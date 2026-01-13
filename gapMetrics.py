import numpy as np                        # For linear algebra
import cvxpy as cp
import matplotlib.pyplot as plt            # For plots
from scipy.signal import ss2tf
# np.random.seed(1)                          # Generate random seed
# np.set_printoptions(precision=1)           # Set nice printing format

def is_controllable(A, B, tol=1e-9):
    """
    Check controllability of (A,B) using the controllability matrix:
        C = [B, AB, A^2 B, ..., A^{n-1} B]
    """
    n = A.shape[0]
    C = B
    for _ in range(1, n):
        C = np.hstack((C, A @ C[:, -B.shape[1]:]))
    rank = np.linalg.matrix_rank(C, tol=tol)
    return rank == n

def is_observable(A, C, tol=1e-9):
    """
    Check observability of (A,C) using the observability matrix:
        O = [C; CA; CA^2; ...; CA^{n-1}]
    """
    n = A.shape[0]
    O = C
    for _ in range(1, n):
        O = np.vstack((O, O[-C.shape[0]:, :] @ A))
    rank = np.linalg.matrix_rank(O, tol=tol)
    return rank == n

def random_discrete_system(n=2, m=1, p=1, max_tries=1000, scale_A=1.0):
    """
    Generate a random discrete-time state-space system:
        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_k

    such that (A,B) is controllable and (A,C) is observable.

    n : number of states
    m : number of inputs
    p : number of outputs
    max_tries : how many random draws before giving up
    scale_A : multiply A by this factor (you can <1.0 to make it "more stable"
    """
    for _ in range(max_tries):
        A = scale_A * np.random.randn(n, n)
        B = np.random.randn(n, m)
        C = np.random.randn(p, n)
        D = np.zeros((p, m))  # often we take D = 0

        if is_controllable(A, B) and is_observable(A, C):
            return A, B, C, D

    raise RuntimeError("Failed to find controllable & observable system in max_tries.")

def ss_to_tf_discrete(A, B, C, D):
    """
    Compute discrete-time transfer function G(z) from (A,B,C,D).

    Returns:
        num: array of numerator coefficient arrays, shape (p, m), each entry is 1D ndarray
        den: 1D ndarray of denominator coefficients (same for all I/O pairs)
    """
    p, m = D.shape
    num = np.empty((p, m), dtype=object)
    den = None

    for j in range(m):
        # ss2tf returns numerator(s) for all outputs for input j
        num_j, den_j = ss2tf(A, B, C, D, input=j)
        if den is None:
            den = den_j
        for i in range(p):
            num[i, j] = num_j[i, :]

    return num, den

#1.2 Generate Hankel matrix
def persistently_exciting_input(T, m=1, std=1.0):
    """
    Generate persistently exciting input (white noise).
    m = number of inputs
    T = length
    """
    return np.random.randn(T, m) * std

def simulate_system(A, B, C, D, u, x0=None):
    """
    Simulate x_{k+1} = A x_k + B u_k,  y_k = C x_k + D u_k

    u : array (T, m)
    """
    A = np.asarray(A); B = np.asarray(B)
    C = np.asarray(C); D = np.asarray(D)

    T, m = u.shape
    n = A.shape[0]
    p = C.shape[0]

    if x0 is None:
        x0 = np.zeros((n, 1))

    x = np.zeros((T+1, n))
    y = np.zeros((T, p))

    x[0] = x0[:,0]

    for k in range(T):
        x[k+1] = A @ x[k] + B @ u[k]
        y[k]   = C @ x[k] + D @ u[k]

    return x, y

def hankel_matrix(data, L):
    """
    Build a Hankel matrix of depth L from a 1D or vector sequence.

    data: array (T, d)  → T time steps, d-dimensional signal (e.g. input or output)
    L: depth
    Returns H ∈ R^{L*d × (T-L+1)}
    """
    data = np.asarray(data)
    T, d = data.shape
    cols = T - L + 1
    H = np.zeros((L*d, cols))
    for i in range(L):
        H[i*d:(i+1)*d, :] = data[i:i+cols].T
    return H

#2.3 principal angles
# Guide for principal angle code https://pyphysim.readthedocs.io/en/latest/_modules/pyphysim/subspace/metrics.html
def principal_angles(H1, H2, return_degrees=False):
    """Compute principal angles between subspaces span(H1) and span(H2).

    H1: (n, p) matrix whose columns span first subspace
    H2: (n, q) matrix whose columns span second subspace
    return_degrees: if True, return angles in degrees instead of radians

    Returns:
        angles: array of principal angles (length = min(rank(H1), rank(H2)))
    """

    # Orthonormal bases via QR
    Q1 = np.linalg.qr(H1)[0]
    Q2 = np.linalg.qr(H2)[0]
    # SVD of Q1^T Q2
    M = Q1.conjugate().transpose().dot(Q2)
    S= np.linalg.svd(M, full_matrices=False)[1]
    # print("Singular values:", S)

    # Clip numerical noise into [-1,1] then take arccos
    S = np.clip(S, -1.0, 1.0)
    # S[S > 1] = 1
    angles = np.arccos(S)  # radians, between 0 and pi/2
    # print("Principal angles (radians):", angles)
    if return_degrees:
        angles = angles * 180 / np.pi

    return angles  # sorted from smallest to largest

def Lgap_metric(A, B):
    """
    Compute gap(U,V) for subspaces U=span(A), V=span(B),
    assuming dim(U) = dim(V) and both have the same rank.

    Returns:
        gap_value, theta_max (in radians)
    """
    angles = principal_angles(A, B)

    # largest principal angle
    theta_max = np.max(angles)

    # gap = sin(theta_max)  (for equal dimensions)
    gap_value = np.sin(theta_max)

    return gap_value, theta_max