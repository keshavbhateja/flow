
import math
import random
import csv
from itertools import product
from collections import Counter, defaultdict

# ======================
# Config (edit as needed)
# ======================

# Number of rounds (4-qubit resource state)
N = 4

# Angle set (in radians)
ANGLE_SET = [math.pi/4, 3*math.pi/4, 5*math.pi/4, 7*math.pi/4]

# Number of Monte Carlo trials per (flow, alpha-sequence) configuration
NUM_TRIALS = 5000  # Increase for tighter estimates

# Output CSV path
OUT_CSV = "joint_probs_empirical_N4.csv"

# ======================
# Helper functions
# ======================

def normalize_angle(theta):
    """Normalize angle to (-π, π] for readability and stable bucketing."""
    two_pi = 2*math.pi
    t = theta % two_pi
    if t > math.pi:
        t -= two_pi
    if abs(t) < 1e-12:
        t = 0.0
    if abs(t - math.pi) < 1e-12:
        t = math.pi
    if abs(t + math.pi) < 1e-12:
        t = -math.pi
    return t

def angle_to_k_pi_over_4(theta):
    """Map an angle to integer k where theta ≈ k * (π/4) after normalization."""
    t = normalize_angle(theta)
    k = int(round((t / math.pi) * 4))
    t2 = normalize_angle(k * (math.pi/4))
    k2 = int(round((t2 / math.pi) * 4))
    return k2

def key_angle_list(angles):
    return tuple(angle_to_k_pi_over_4(a) for a in angles)

def parity(bits):
    """Return XOR of all bits in the list (0 if empty)."""
    p = 0
    for b in bits:
        p ^= (b & 1)
    return p

# ======================
# Flow definitions (9 variants; placeholders to be replaced with exact protocol flows)
# Each flow provides s_x(j, b_prefix) and s_z(j, b_prefix) for j=1..N.
# b_prefix contains decoded bits [b1, ..., b_{j-1}].
# ======================

def b_at(prefix, idx_zero_based):
    return prefix[idx_zero_based] if len(prefix) > idx_zero_based else 0

def flow_builder(sx_rule, sz_rule):
    def s_x(j, b_prefix):
        if j == 1:  # typical MBQC: no correction before first outcome
            return 0
        return sx_rule(j, b_prefix) & 1
    def s_z(j, b_prefix):
        if j == 1:
            return 0
        return sz_rule(j, b_prefix) & 1
    return s_x, s_z

# Define a few dependency patterns
def sx_const0(j, b_pref): return 0
def sx_const1(j, b_pref): return 1
def sx_b1(j, b_pref):    return b_at(b_pref, 0)
def sx_b2(j, b_pref):    return b_at(b_pref, 1)
def sx_b3(j, b_pref):    return b_at(b_pref, 2)
def sx_parity(j, b_pref):return parity(b_pref)
def sx_b1_xor_b3(j,b_pref): return (b_at(b_pref,0) ^ b_at(b_pref,2))

# Reuse same shapes for sz
sz_const0 = sx_const0
sz_const1 = sx_const1
sz_b1     = sx_b1
sz_b2     = sx_b2
sz_b3     = sx_b3
sz_parity = sx_parity
sz_b1_xor_b3 = sx_b1_xor_b3

FLOW_RULES = {
    1: flow_builder(sx_const0,      sz_const0),
    2: flow_builder(sx_b1,          sz_const0),
    3: flow_builder(sx_const0,      sz_b1),
    4: flow_builder(sx_b1,          sz_b1),
    5: flow_builder(sx_parity,      sz_const0),
    6: flow_builder(sx_const0,      sz_parity),
    7: flow_builder(sx_b2,          sz_const0),
    8: flow_builder(sx_const0,      sz_b2),
    9: flow_builder(sx_b1_xor_b3,   sz_parity),
}

# ======================
# Core protocol functions
# ======================

def compute_alpha_prime(alpha_j, j, s_x, s_z, b_prefix, r_j):
    """α'_j = (-1)^{s^x_j} * α_j + π * (s^z_j ⊕ r_j)  (mod 2π)."""
    sx = s_x(j, b_prefix) & 1
    sz = s_z(j, b_prefix) & 1
    theta = ((-1)**sx) * alpha_j + math.pi * (sz ^ (r_j & 1))
    return normalize_angle(theta)

def honest_born_prob_zero(alpha_prime_j):
    """
    Honest-quantum toy model for measurement outcome:
      P(b'_j = 0 | α'_j) = cos^2(α'_j / 2)
      P(b'_j = 1 | α'_j) = sin^2(α'_j / 2)
    """
    return math.cos(alpha_prime_j / 2.0) ** 2

def simulate_once(alpha_vec, flow_id, rng):
    """Run one Monte Carlo sample, returning (bp, a_prime, r, b)."""
    s_x, s_z = FLOW_RULES[flow_id]

    # Flip random one-time-pad bits r
    r = [rng.randint(0,1) for _ in range(N)]

    # Iteratively compute α' and sample b'
    a_prime = []
    bp = []
    b_decoded = []

    for j in range(1, N+1):
        b_prefix = b_decoded[:]

        a_jp = compute_alpha_prime(alpha_vec[j-1], j, s_x, s_z, b_prefix, r[j-1])
        a_prime.append(a_jp)

        # Sample b'_j from the honest model
        p0 = honest_born_prob_zero(a_jp)
        u = rng.random()
        bpj = 0 if u < p0 else 1
        bp.append(bpj)

        # Decode b_j
        bj = bpj ^ r[j-1]
        b_decoded.append(bj)

    return bp, a_prime, r, b_decoded

# ======================
# Joint probability estimation
# ======================

def simulate(flow_ids=range(1,10), angle_set=ANGLE_SET, num_trials=NUM_TRIALS, seed=12345):
    """
    Monte Carlo estimate of the joint distribution over
      (flow_id, alpha, alpha_prime, r, bprime, b)
    with α chosen exhaustively from angle_set^N, r random fair coins,
    and b' sampled from the honest model.
    """
    rng = random.Random(seed)

    # Build all α vectors from the angle set
    all_alphas = list(product(angle_set, repeat=N))

    counts = Counter()
    totals_for_config = defaultdict(int)  # count per (flow_id, alpha)

    for flow_id in flow_ids:
        for alpha_vec in all_alphas:
            for _ in range(num_trials):
                bp, a_prime, r, b = simulate_once(alpha_vec, flow_id, rng)

                # Hashable keys (quantize angles to k*pi/4 for stable grouping)
                alpha_key      = key_angle_list(alpha_vec)
                alpha_p_key    = key_angle_list(a_prime)
                r_key          = tuple(r)
                bp_key         = tuple(bp)
                b_key          = tuple(b)

                key = (flow_id, alpha_key, alpha_p_key, r_key, bp_key, b_key)
                counts[key] += 1
                totals_for_config[(flow_id, alpha_key)] += 1

    # Convert to probabilities per (flow, alpha) configuration
    rows = []
    for key, c in counts.items():
        flow_id, alpha_key, alpha_p_key, r_key, bp_key, b_key = key
        Z = totals_for_config[(flow_id, alpha_key)]
        p = c / Z if Z > 0 else 0.0
        rows.append({
            "flow_id": flow_id,
            "alpha_k*pi/4": alpha_key,
            "alpha_prime_k*pi/4": alpha_p_key,
            "r": r_key,
            "bprime": bp_key,
            "b": b_key,
            "count": c,
            "prob": p
        })

    # Sort rows for a consistent CSV
    rows.sort(key=lambda d: (d["flow_id"], d["alpha_k*pi/4"], d["alpha_prime_k*pi/4"], d["r"], d["bprime"], d["b"]))

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flow_id","alpha_k*pi/4","alpha_prime_k*pi/4","r","bprime","b","count","prob"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Saved empirical joint table to: {OUT_CSV}")
    print("Preview of first 10 rows:")
    for r_ in rows[:10]:
        print(r_)

    return rows

if __name__ == "__main__":
    simulate()
