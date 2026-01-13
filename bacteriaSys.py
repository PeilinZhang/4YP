import numpy as np
from scipy.linalg import expm

IDX = {
    "m_t":0, "m_m":1, "m_q":2, "m_z":3, "m_g":4,
    "M_t":5, "M_m":6, "M_q":7, "M_z":8, "M_g":9,
    "p_t":10,"p_m":11,"p_q":12,"p_z":13,"p_g":14,
    "P_g":15,
    "s":16,
    "a":17
}

HOST_GENES = ["t","m","q","z"]
ALL_GENES  = ["t","m","q","z","g"]

def unpack(x):
    """Convenience accessor."""
    d = {}
    for k,i in IDX.items():
        d[k] = x[i]
    return d

def default_params():
    params = {} #parameters

    # Transcription (Eq. 3): alpha_x,max, theta_x; plus q autoregulation Aq,hq
    for gene in ["t","m","z"]:
        params[f"alpha_{gene}_max"] = 1.0
        params[f"theta_{gene}"] = 100.0

    params["alpha_q_max"] = 1.0
    params["theta_q"] = 100.0
    params["Aq"] = 1e4
    params["hq"] = 2.0

    # Translation elongation gamma(a) (Eq. 4)
    params["gamma_max"] = 20.0   # aa/min
    params["K_gamma"] = 100.0
    params["rho"] = 1e6          # aa per cell (placeholder)
    #also the free ribosome and mRNA binding rates are with Eq. 6

    # protein lengths (aa)
    params["n_t"] = 300
    params["n_m"] = 300
    params["n_q"] = 300
    params["n_z"] = 7000
    params["n_g"] = 240

    # mRNA decay delta_x (Eq. 6)
    for gene in ALL_GENES:
        params[f"delta_{gene}"] = 0.01
        params[f"kplus_{gene}"] = 1e-4
        params[f"kminus_{gene}"] = 1e-2

    # Nutrient/energy dynamics (Eq. 7)
    params["Vt"] = 1.0
    params["At"] = 1.0
    params["Vm"] = 1.0
    params["Am"] = 1.0
    params["eta_s"] = 1.0

    # Synthetic transcription (Eq. 8)
    params["alpha_syn_max"] = 1.0
    params["theta_syn"] = 100.0
    params["Fb"] = 0.01
    params["Ag"] = 1.0
    params["hg"] = 2.0
    # delay tau_g exists in paper; ignored in this algebraic f(x,u) skeleton

    # GFP maturation (Eq. 9): mu_g
    params["mu_g"] = 0.01

    return params

def gamma(a, params):
    # Eq. 4: gamma(a) = (gamma_max * a)/(K_gamma + a)
    # a is the energy bearing molecule (e.g ATP and NADPH)
    # gamma is translation elongation rate in aa/min
    return (params["gamma_max"] * a) / (params["K_gamma"] + a + 1e-12)

def v_of_gene(a, gene, params):
    # Eq. 4: v_x(a) = gamma(a) / n_x
    return gamma(a, params) / params[f"n_{gene}"]

def alpha_host(a, gene, pq, params):
    # Eq. 3a and 3b
    if gene == "q":
        return (params["alpha_q_max"] * a) / (params["theta_q"] + a + 1e-12) * (1.0 / (1.0 + (pq/(params["Aq"]+1e-12))**params["hq"]))
    else:
        return (params[f"alpha_{gene}_max"] * a) / (params[f"theta_{gene}"] + a + 1e-12)

def alpha_syn(a, ug, p):
    # Eq. (8) but ignoring delay ug(t - tau_g) doe simplicity f(x,u)
    hill = (ug**p["hg"]) / (p["Ag"] + ug**p["hg"] + 1e-12)
    return (p["alpha_syn_max"] * a) / (p["theta_syn"] + a + 1e-12) * (p["Fb"] + hill)

def h_function(x, u, p):
    """Outputs y = h(x): [growth rate lambda, mature GFP P_g]."""
    # u is not used but included
    d = unpack(x)
    # Eq. (5)
    gamma_val = gamma(d["a"], p)
    M_sum = d["M_t"] + d["M_m"] + d["M_q"] + d["M_z"] + d["M_g"]
    lam = (gamma_val / p["rho"]) * M_sum

    return np.array([lam, d["P_g"]], dtype=float)

def f_function(x, u, p=None):
    """
    Continuous-time dynamics xdot = f(x,u) using Eqs (6)-(7)-(8)-(9).
    u = [u_s, u_g]
    not normalised
    """
    if p is None:
        p = default_params()

    d = unpack(x)
    us, ug = float(u[0]), float(u[1])

    # states
    a = d["a"]
    s = d["s"]
    pz = d["p_z"]  # free ribosomes
    pq = d["p_q"]

    # rates
    gamma_value = gamma(a, p)
    v = {gene: v_of_gene(a, gene, p) for gene in ALL_GENES}

    # growth rate lambda (Eq. 5)
    M_sum = d["M_t"] + d["M_m"] + d["M_q"] + d["M_z"] + d["M_g"]
    lam = (gamma_value / p["rho"]) * M_sum

    # transcription
    alpha = {}
    for gene in HOST_GENES:
        alpha[gene] = alpha_host(a, gene, pq, p)
    alpha["g"] = alpha_syn(a, ug, p)

    xdot = np.zeros_like(x, dtype=float)

    # Eq. (6a)-(6b) for each gene in ALL_GENES
    for gene in ALL_GENES:
        m = d[f"m_{gene}"]
        M = d[f"M_{gene}"]
        delta = p[f"delta_{gene}"]
        kplus = p[f"kplus_{gene}"]
        kminus = p[f"kminus_{gene}"]

        # m_dot
        m_dot = alpha[gene] - (lam + delta + kplus * pz) * m + (v[gene] + kminus) * M
        # M_dot
        M_dot = -(lam + v[gene] + kminus) * M + (kplus * pz) * m

        xdot[IDX[f"m_{gene}"]] = m_dot
        xdot[IDX[f"M_{gene}"]] = M_dot

    # Eq. (6c) for proteins p_t,p_m,p_q and p_g
    for gene in ["t","m","q","g"]:
        M = d[f"M_{gene}"]
        p_gene = d[f"p_{gene}"]
        xdot[IDX[f"p_{gene}"]] = v[gene] * M - lam * p_gene

    # Ribosomes: Eq. (6d)
    # TODO To avoid double-counting vzMz, we assume the summation is over S \ {z}.
    term_new_rib = v["z"] * d["M_z"]
    term_release = 0.0
    term_bind = 0.0
    term_dissoc = 0.0
    for gene in ALL_GENES:  # TODO S \ {z}
        term_release += v[gene] * d[f"M_{gene}"]
        term_bind    += p[f"kplus_{gene}"] * d[f"m_{gene}"] * pz
        term_dissoc  += p[f"kminus_{gene}"] * d[f"M_{gene}"]

    xdot[IDX["p_z"]] = term_new_rib - lam * pz + (term_release - term_bind + term_dissoc)

    # Nutrient/energy: Eq. (7a)-(7b)
    # s_dot = p_t * Vt*us/(At+us) - p_m * Vm*s/(Am+s) - lam*s
    s_dot = d["p_t"] * (p["Vt"] * us / (p["At"] + us + 1e-12)) \
          - d["p_m"] * (p["Vm"] * s  / (p["Am"] + s  + 1e-12)) \
          - lam * s
    xdot[IDX["s"]] = s_dot

    # a_dot = eta_s * p_m * Vm*s/(Am+s) - lam*a - sum_{x in S} gamma(a)*M_x
    a_prod = p["eta_s"] * d["p_m"] * (p["Vm"] * s / (p["Am"] + s + 1e-12))
    a_cons = (gamma_value * (d["M_t"] + d["M_m"] + d["M_q"] + d["M_z"] + d["M_g"]))
    xdot[IDX["a"]] = a_prod - lam * a - a_cons

    # GFP maturation: Eq. (9)
    # ṗg already set above as v_g*M_g - lam*p_g; but Eq. (9) adds -mu_g*p_g
    xdot[IDX["p_g"]] -= p["mu_g"] * d["p_g"]
    # Ṗg = mu_g*p_g - lam*P_g
    xdot[IDX["P_g"]] = p["mu_g"] * d["p_g"] - lam * d["P_g"]

    return xdot

def finite_difference_jacobian_fx(f, x0, u0, p, eps=1e-6):
    """Return A = df/dx using central differences."""
    nx = x0.size
    A = np.zeros((nx, nx))
    for j in range(nx):
        dx = np.zeros(nx)
        step = eps * max(1.0, abs(x0[j])) #for numerical stability
        dx[j] = step
        fp = f(x0 + dx, u0, p)
        fm = f(x0 - dx, u0, p)
        A[:, j] = (fp - fm) / (2.0 * step)
    return A

def finite_difference_jacobian_fu(f, x0, u0, p, eps=1e-6):
    """Return B = df/du using central differences."""
    nu = u0.size
    nx = x0.size
    B = np.zeros((nx, nu))
    for j in range(nu):
        du = np.zeros(nu)
        step = eps * max(1.0, abs(u0[j]))
        du[j] = step
        fp = f(x0, u0 + du, p)
        fm = f(x0, u0 - du, p)
        B[:, j] = (fp - fm) / (2.0 * step)
    return B

def finite_difference_jacobian_hx(h, x0, u0, p, eps=1e-6):
    """Return C = dh/dx using central differences."""
    y0 = h(x0, u0, p)
    ny = y0.size
    nx = x0.size
    C = np.zeros((ny, nx))
    for j in range(nx):
        dx = np.zeros(nx)
        step = eps * max(1.0, abs(x0[j]))
        dx[j] = step
        hp = h(x0 + dx, u0, p)
        hm = h(x0 - dx, u0, p)
        C[:, j] = (hp - hm) / (2.0 * step)
    return C


def zoh(A, B, Ts):
    """
    Discretise xdot = A x + B u under ZOH with sampling time Ts:
      Ad = expm(A Ts)
      Bd = integral_0^Ts expm(A tau) B d tau
    Using block-matrix exponential (robust even if A is singular).
    """
    nx, nu = B.shape
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    Md = expm(M * Ts)
    Ad = Md[:nx, :nx]
    Bd = Md[:nx, nx:] #integration is computed implicitly
    return Ad, Bd

#make a function that automatically outputs the linearised discrete-time system
def linearised_discrete_system(x0, u0, Ts, p=None):
    """Returns (Ad,Bd,Cd) linearised about (x0,u0) with sampling time Ts."""
    if p is None:
        p = default_params()

    A = finite_difference_jacobian_fx(f_function, x0, u0, p)
    B = finite_difference_jacobian_fu(f_function, x0, u0, p)
    C = finite_difference_jacobian_hx(h_function, x0, u0, p)

    Ad, Bd = zoh(A, B, Ts)
    Cd = C  # no discretisation needed for output matrix

    return Ad, Bd, Cd

#make a function that changes a number of parameters
def set_params(**kwargs):
    """Set multiple parameters in the params dictionary."""
    p = default_params()
    for key, value in kwargs.items():
        if key in p:
            p[key] = value
        else:
            raise KeyError(f"Parameter '{key}' not found in parameters dictionary.")
    return p
