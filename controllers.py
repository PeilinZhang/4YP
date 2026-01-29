import cvxpy as cp
import numpy as np

def DataLQRcontroller(U0, X0, X1, Qx, R):
    n = X0.shape[0]
    m = U0.shape[0]
    T = X0.shape[1]
    Q = cp.Variable((T,n))
    X = cp.Variable((m,m), PSD=True)    # Precompute matrices used inside the LMIs
        
    R12 = np.linalg.cholesky(R)              # R^{1/2}
    X0Q = X0 @ Q                             # (n×T)(T×n) = (n×n)
    X1Q = X1 @ Q                             # (n×n)
    U0Q = U0 @ Q                             # (m×T)(T×n) = (m×n)

    blk1 = cp.bmat([
        [X,                R12 @ U0Q],
        [(U0Q).T@R12,      X0Q      ]
    ])

    blk2 = cp.bmat([
        [X0Q - np.eye(n),  X1Q],
        [X1Q.T,            X0Q]
    ])

    # cost function
    cost = cp.trace(Qx @ X0Q) + cp.trace(X)

    # set the DeePC equality constraints
    constraints = [blk1 >> 0, blk2 >> 0]
        
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS,
        verbose=False,
        max_iters=20000,        # raise from default
        eps=1e-5,
        warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP solve failed: {prob.status}")
    
    X0Q_val = (X0 @ Q).value          # (n×n)
    K = -(U0 @ Q.value) @ np.linalg.inv(X0Q_val)  # (m×n)
    return K