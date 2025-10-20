import numpy as np
from itertools import product
from math import log2, pi

def tensor_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

I = np.eye(2, dtype=complex)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)

def CZ_on(n, a, b):
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for idx in range(dim):
        if ((idx >> a) & 1) and ((idx >> b) & 1):
            U[idx, idx] *= -1
    return U

def plus_state(n):
    v = np.zeros((2**n,), dtype=complex)
    v[0] = 1.0
    for q in range(n):
        ops = [I]*n
        ops[q] = H
        Uq = tensor_list(ops)
        v = Uq @ v
    return v

def projector_plus_minus_alpha(alpha, outcome):
    phase = np.exp(1j*alpha)
    ket = np.array([1.0, phase if outcome==0 else -phase], dtype=complex)/np.sqrt(2)
    return np.outer(ket, ket.conj())

def apply_projective_measurement(state, n, qubit, alpha, outcome):
    P = projector_plus_minus_alpha(alpha, outcome)
    ops = [I]*n
    ops[qubit] = P
    M = tensor_list(ops)
    post = M @ state
    prob = np.vdot(post, post).real
    if prob > 0:
        post = post / np.sqrt(prob)
    return post, prob

def cluster_state_2x2():
    n = 4
    psi = plus_state(n)
    for (a,b) in [(0,1),(0,2),(1,3),(2,3)]:
        psi = CZ_on(n, a, b) @ psi
    return psi

def distribution_b_given_alpha_prime(alpha_tuple):
    n = 4
    psi0 = cluster_state_2x2()
    branches = [(psi0, 1.0, ())]
    for q in range(n):
        new = []
        for (state, p_prefix, bits) in branches:
            for b in (0,1):
                post, p = apply_projective_measurement(state, n, q, alpha_tuple[q], b)
                new.append((post, p_prefix*p, bits+(b,)))
        branches = new
    dist = {}
    for (_, p, bits) in branches:
        dist[bits] = dist.get(bits, 0.0) + p
    total = sum(dist.values())
    for k in dist:
        dist[k] /= total if total>0 else 1.0
    return dist

def entropy_from_dist(dist):
    return -sum(p*log2(p) for p in dist.values() if p>0)

def compute_entropies_for_angle_set(angle_set):
    alpha_primes = list(product(angle_set, repeat=4))
    H_B_given_A = 0.0
    for alpha_tuple in alpha_primes:
        H_B_given_A += entropy_from_dist(distribution_b_given_alpha_prime(alpha_tuple))
    H_B_given_A /= len(alpha_primes)

    H_A = 4 * log2(len(angle_set))
    H_BA = H_A + H_B_given_A

    N = 4
    H_BA_given_AF = H_B_given_A + N

    n_F = log2(9)              # 9 valid flows on 2x2
    H_AF = H_A + n_F
    MI = H_BA - H_BA_given_AF
    CE = H_AF - MI
    return H_B_given_A, H_A, H_BA, H_BA_given_AF, MI, CE

A4 = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]
A2 = [0.0, pi]
labels = ["H(B'|A')","H(A')","H(B',A')","H(B',A'|A,F)","MI","CE"]

print("=== A4 ===")
print(dict(zip(labels, [f"{v:.10f}" for v in compute_entropies_for_angle_set(A4)])))

print("=== A2 ===")
print(dict(zip(labels, [f"{v:.10f}" for v in compute_entropies_for_angle_set(A2)])))

