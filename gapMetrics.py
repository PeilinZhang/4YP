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

def eval_tf(num, den, z):
    """
    Evaluate a MIMO transfer function matrix P(z) at a complex point z.

    num: array of shape (p, m), each entry is a 1D array of numerator coeffs
    den: 1D array of denominator coeffs
    z: complex scalar (e^{jω} in discrete-time)

    Returns: P(z) as a (p, m) complex ndarray.
    """
    p, m = num.shape
    P = np.zeros((p, m), dtype=complex)
    den_val = np.polyval(den, z)
    for i in range(p):
        for j in range(m):
            P[i, j] = np.polyval(num[i, j], z) / den_val
    return P

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

#for v gap
def inv_sqrt_hermitian(M, eps=1e-12):
    lam, V = np.linalg.eigh(M)
    lam_clipped = np.clip(lam, eps, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(lam_clipped))
    return V @ D_inv_sqrt @ V.conj().T

def L_mat(P1,P2):
    p, m = P1.shape
    I_p = np.eye(p, dtype=complex)
    return inv_sqrt_hermitian(I_p + P1 @ P2.conj().T)

def R_mat(P1,P2):
    p, m = P1.shape
    I_m = np.eye(m, dtype=complex)
    return inv_sqrt_hermitian(I_m + P1.conj().T @ P2)

def G_star_G(P1, P2):
    p, m = P1.shape
    R1 = R_mat(P1, P1)
    R2 = R_mat(P2, P2)
    middle = np.eye(m, dtype=complex) + P2.conj().T @ P1
    return R1.conj().T @ middle @ R2

def winding_number(det_vals):
    # Argument at each sample
    angles = np.unwrap(np.angle(det_vals))
    # Net change in angle over the loop
    dtheta = angles[-1] - angles[0]
    return int(np.round(dtheta / (2 * np.pi)))

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

def vgap_metric(num1, den1, num2, den2):
    """
    Compute v-gap
    """
    n_freq = 1000
    eps_det=1e-6
    omegas = np.linspace(-np.pi, np.pi, n_freq)
    v_r = 0.0
    det_vals = []
    for w in omegas:
        z = np.exp(1j * w)
        P1 = eval_tf(num1, den1, z)
        P2 = eval_tf(num2, den2, z)
        p, m = P1.shape
        # --- a_v part: || L(P2) (P2 - P1) R(P1) ||_2 ---
        L2 = L_mat(P2,P2)
        R1 = R_mat(P1,P1)
        M = L2 @ (P2 - P1) @ R1   # p x m
        # spectral norm (largest singular value)
        v_r = max(v_r, np.linalg.norm(M, 2))
        # --- winding-number part: det(G1^* G2) ---
        G1G2_star = G_star_G(P1, P2)
        det_val = np.linalg.det(G1G2_star)
        det_vals.append(det_val)
    det_vals = np.array(det_vals)
    # Check det != 0 everywhere on the grid
    if np.min(np.abs(det_vals)) < eps_det:
        return 1.0, v_r
    # Approximate winding number of det(G1^* G2)
    wno = winding_number(det_vals)
    if wno != 0:
        return 1.0, v_r
    return float(v_r), v_r

#for bound
#find v gap boundary, controller transfer function is needed
def vgap_bpc(P_num, P_den, C_num, C_den):
    """
    Approximate b_{P,C} = 1 / || [P; I] (I - C P)^(-1) [-C  I] ||_inf
    by frequency sampling on the unit circle.
    P is the transfer function of the plant,
    C is the transfer function of the controller.

    Returns:
      b_pc, hinf_approx, w_peak
    """
    n_freq = 2000
    eps=1e-12
    p, m = P_num.shape
    mC_out, pC_in = C_num.shape
    if (mC_out, pC_in) != (m, p):
        raise ValueError(f"Expected C to be shape (m,p)=({m},{p}), got {C_num.shape}")
    I_m = np.eye(m, dtype=complex)
    # frequency grid
    ws = np.linspace(0, np.pi, n_freq)  # discrete-time: [0, pi] is enough
    peak_sigma = -np.inf
    w_peak = None
    for w in ws:
        z = np.exp(1j * w)
        P = eval_tf(P_num, P_den, z)   # (p,m)
        C = eval_tf(C_num, C_den, z)   # (m,p)
        # Compute S = (I - C P)^(-1)
        Mmid = I_m - C @ P
        # protect against near-singular frequency points
        if np.linalg.cond(Mmid) > 1/eps:
            # treat as extremely large gain -> dominates hinf
            sigma_max = np.inf
        else:
            S = np.linalg.inv(Mmid)  # (m,m)

            # Build blocks: [P; I_m] and [-C  I_m]
            left = np.vstack([P, I_m])                 # (p+m, m)
            right = np.hstack([-C, I_m])               # (m, p+m)

            M = left @ S @ right                       # (p+m, p+m)

            # largest singular value
            sigma_max = np.linalg.svd(M, compute_uv=False)[0]
        if sigma_max > peak_sigma:
            peak_sigma = sigma_max
            w_peak = w
    hinf_approx = peak_sigma
    b_pc = 1.0 / hinf_approx if np.isfinite(hinf_approx) and hinf_approx > 0 else 0.0
    return b_pc, hinf_approx, w_peak