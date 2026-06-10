import numpy as np

eps = 1e-300

def _newton(p0, p1, grid_size=16, time_steps=20,
            beta=1e-3, tol=1e-5, max_iter=50, alpha0=0.3,
            eps_weight=1.0, w_x_floored=None, w_y_floored=None):

    
    from scipy.sparse import lil_matrix, bmat
    from scipy.sparse.linalg import spsolve

    N = int(grid_size)
    L = time_steps
    dt = 1.0 / (L + 1)
    T = L + 1
    x = np.linspace(0.0, 1.0, grid_size)
    dx = x[1] - x[0]
    inv_dx2 = 1.0 / (dx**2)
    C = (beta**2) * inv_dx2

    def f_grad_hess(u):
        m_x, m_y, p_int = unpack_u(u)
        p = assemble_p(p_int)

        if np.min(p[1:-1]) <= 0.0:
            return np.inf, None, None

        g = np.zeros(nvar, dtype=float)
        H = lil_matrix((nvar, nvar), dtype=float)

        f = 0.0

        for t in range(T):
            pt = p[t]
            log_pt = np.log(pt)

            for i in range(N):
                for j in range(N - 1):
                    mx_i = idx_mx(t, i, j)
                    m = m_x[t, i, j]
                    pa = pt[i, j]
                    pb = pt[i, j + 1]
                    gg = 0.5 * (pa + pb)

                    f += (m**2) / gg

                    g[mx_i] += 2.0 * m / gg
                    H[mx_i, mx_i] += 2.0 / gg

                    ia = idx_p(t, i, j)
                    ib = idx_p(t, i, j + 1)

                    gpa = -(m**2) / (2.0 * gg**2)
                    hmp = -m / (gg**2)
                    hpp = (m**2) / (2.0 * gg**3)

                    if ia is not None:
                        g[ia] += gpa
                        H[mx_i, ia] += hmp
                        H[ia, mx_i] += hmp
                        H[ia, ia] += hpp
                    if ib is not None:
                        g[ib] += gpa
                        H[mx_i, ib] += hmp
                        H[ib, mx_i] += hmp
                        H[ib, ib] += hpp
                    if ia is not None and ib is not None:
                        H[ia, ib] += hpp
                        H[ib, ia] += hpp

                    w = w_x_floored[t, i, j]
                    Cw = C * w
                    
                    s = log_pt[i, j] - log_pt[i, j + 1]
                    f += Cw * (s**2) * gg
                    
                    ga = Cw * (2.0 * s * gg / pa + 0.5 * s**2)
                    gb = Cw * (-2.0 * s * gg / pb + 0.5 * s**2)
                    
                    haa = Cw * (2.0 * gg / (pa * pa) + 2.0 * s / pa - 2.0 * s * gg / (pa * pa))
                    hbb = Cw * (2.0 * gg / (pb * pb) - 2.0 * s / pb + 2.0 * s * gg / (pb * pb))
                    hab = Cw * (-2.0 * gg / (pa * pb) + s / pa - s / pb)

                    if ia is not None:
                        g[ia] += ga
                        H[ia, ia] += haa
                    if ib is not None:
                        g[ib] += gb
                        H[ib, ib] += hbb
                    if ia is not None and ib is not None:
                        H[ia, ib] += hab
                        H[ib, ia] += hab

            for i in range(N - 1):
                for j in range(N):
                    my_i = idx_my(t, i, j)
                    m = m_y[t, i, j]
                    pa = pt[i, j]
                    pb = pt[i + 1, j]
                    gg = 0.5 * (pa + pb)

                    f += (m**2) / gg

                    g[my_i] += 2.0 * m / gg
                    H[my_i, my_i] += 2.0 / gg

                    ia = idx_p(t, i, j)
                    ib = idx_p(t, i + 1, j)

                    gpa = -(m**2) / (2.0 * gg**2)
                    hmp = -m / (gg**2)
                    hpp = (m**2) / (2.0 * gg**3)

                    if ia is not None:
                        g[ia] += gpa
                        H[my_i, ia] += hmp
                        H[ia, my_i] += hmp
                        H[ia, ia] += hpp
                    if ib is not None:
                        g[ib] += gpa
                        H[my_i, ib] += hmp
                        H[ib, my_i] += hmp
                        H[ib, ib] += hpp
                    if ia is not None and ib is not None:
                        H[ia, ib] += hpp
                        H[ib, ia] += hpp

                    w = w_y_floored[t, i, j]
                    Cw = C * w
                    
                    s = log_pt[i, j] - log_pt[i + 1, j]
                    f += Cw * (s**2) * gg
                    
                    ga = Cw * (2.0 * s * gg / pa + 0.5 * s**2)
                    gb = Cw * (-2.0 * s * gg / pb + 0.5 * s**2)
                    
                    haa = Cw * (2.0 * gg / (pa * pa) + 2.0 * s / pa - 2.0 * s * gg / (pa * pa))
                    hbb = Cw * (2.0 * gg / (pb * pb) - 2.0 * s / pb + 2.0 * s * gg / (pb * pb))
                    hab = Cw * (-2.0 * gg / (pa * pb) + s / pa - s / pb)


                    if ia is not None:
                        g[ia] += ga
                        H[ia, ia] += haa
                    if ib is not None:
                        g[ib] += gb
                        H[ib, ib] += hbb
                    if ia is not None and ib is not None:
                        H[ia, ib] += hab
                        H[ib, ia] += hab

        return f, g, H.tocsr()

    def idx_mx(t, i, j):
        return t * (N * (N - 1)) + i * (N - 1) + j
        
    def idx_my(t, i, j):
        return nx + t * ((N - 1) * N) + i * N + j

    def idx_p(t, i, j):
        if 1 <= t <= T - 1:
            return idx_p0 + (t - 1) * (N**2) + i * N + j
        return None
        
    def pack_u(m_x, m_y, p_int):
        return np.concatenate([m_x.reshape(-1), m_y.reshape(-1), p_int.reshape(-1)])

    def unpack_u(u):
        m_x = u[:nx].reshape(T, N, N - 1)
        m_y = u[nx:nx + ny].reshape(T, N - 1, N)
        p_int = u[nx + ny:].reshape(T - 1, N, N)
        return m_x, m_y, p_int

    def assemble_p(p_int):
        return np.concatenate([p0[None, :, :], p_int, p1[None, :, :]], axis=0)

    def _neumann_laplacian_2d(N):
        M = N**2
        L = lil_matrix((M, M), dtype=float)
        
        def idx(i, j):
            return i * N + j
        
        for i in range(N):
            for j in range(N):
                k = idx(i, j)
                deg = 0
                if i > 0:
                    L[k, idx(i - 1, j)] = -1.0
                    deg += 1
                if i < N - 1:
                    L[k, idx(i + 1, j)] = -1.0
                    deg += 1
                if j > 0:
                    L[k, idx(i, j - 1)] = -1.0
                    deg += 1
                if j < N - 1:
                    L[k, idx(i, j + 1)] = -1.0
                    deg += 1
                L[k, k] = float(deg)
        
        L[0, :] = 0.0
        L[0, 0] = 1.0
        return L.tocsr()
    
    def build_A():
        A = lil_matrix((ncon, nvar), dtype=float)
        b = np.zeros(ncon, dtype=float)

        row = 0
        inv_dt = 1.0 / dt
        inv_dx = 1.0 / dx

        for t in range(T):
            for i in range(N):
                for j in range(N):
                    if t == 0:
                        A[row, idx_p(1, i, j)] += inv_dt
                        b[row] = p0[i, j] * inv_dt
                    elif t == T - 1:
                        A[row, idx_p(T - 1, i, j)] += -inv_dt
                        b[row] = -p1[i, j] * inv_dt
                    else:
                        A[row, idx_p(t + 1, i, j)] += inv_dt
                        A[row, idx_p(t, i, j)] += -inv_dt

                    if j == 0:
                        A[row, idx_mx(t, i, 0)] += inv_dx
                    elif j == N - 1:
                        A[row, idx_mx(t, i, N - 2)] += -inv_dx
                    else:
                        A[row, idx_mx(t, i, j)] += inv_dx
                        A[row, idx_mx(t, i, j - 1)] += -inv_dx

                    if i == 0:
                        A[row, idx_my(t, 0, j)] += inv_dx
                    elif i == N - 1:
                        A[row, idx_my(t, N - 2, j)] += -inv_dx
                    else:
                        A[row, idx_my(t, i, j)] += inv_dx
                        A[row, idx_my(t, i - 1, j)] += -inv_dx

                    row += 1
        
        return A.tocsr(), b

    if w_x_floored is None:
        w_x_floored = np.ones((T, N, N - 1), dtype=float)
    else:
        w_x_floored = np.asarray(w_x_floored, dtype=float)
        if w_x_floored.shape != (T, N, N - 1):
            raise ValueError(f"w_x_floored must have shape {(T, N, N-1)}, got {w_x_floored.shape}.")
        if not np.isfinite(w_x_floored).all() or (w_x_floored < 0).any():
            raise ValueError("w_x_floored must be finite and nonnegative.")
    
    if w_y_floored is None:
        w_y_floored = np.ones((T, N - 1, N), dtype=float)
    else:
        w_y_floored = np.asarray(w_y_floored, dtype=float)
        if w_y_floored.shape != (T, N - 1, N):
            raise ValueError(f"w_y_floored must have shape {(T, N-1, N)}, got {w_y_floored.shape}.")
        if not np.isfinite(w_y_floored).all() or (w_y_floored < 0).any():
            raise ValueError("w_y_floored must be finite and nonnegative.")


    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    
    if p0.shape != (N, N) or p1.shape != (N, N):
        raise ValueError(f"p0 and p1 must have shape ({N},{N}).")
    
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        raise ValueError("p0 and p1 must be finite arrays.")
    
    if (p0 < 0).any() or (p1 < 0).any():
        raise ValueError("p0 and p1 must be nonnegative (density).")
    
    p0 = np.maximum(p0, eps)
    p1 = np.maximum(p1, eps)

    m0 = float(p0.sum())
    m1 = float(p1.sum())
    if abs(m0 - m1) > 1e-12 * max(1.0, abs(m0), abs(m1)):
        raise ValueError(
            f"Endpoint masses differ after positivity floor (sum(p0)={m0:.6e}, sum(p1)={m1:.6e}). "
            "Normalize p0 and p1 to equal total mass before calling _newton."
        )

    nx  = T * N * (N - 1)
    ny = T * (N - 1) * N
    idx_p0 = nx + ny

    nvar = nx + ny + (T - 1) * (N**2)
    ncon = T * (N**2)

    A, b = build_A()

    p_traj = np.array([(1.0 - (t * dt)) * p0 + (t * dt) * p1 for t in range(T + 1)], dtype=float)
    p_int = (p_traj[1:-1].copy())
    
    Lap = _neumann_laplacian_2d(N)
    
    m_x = np.zeros((T, N, N-1), dtype=float)
    m_y = np.zeros((T, N-1, N), dtype=float)


    for t in range(T):
        rhs = dx**2 * (p_traj[t + 1] - p_traj[t]) / dt

        if abs(rhs.sum()) > 1e-10:
            raise RuntimeError(
                f"Initialization RHS does not satisfy mass conservation at t={t}: sum={rhs.sum():.3e}."
            )
        
        bphi = rhs.reshape(-1).astype(float)
        bphi[0] = 0.0
        Phi = spsolve(Lap, bphi).reshape(N, N)

        m_x[t] = (Phi[:, 1:] - Phi[:, :-1]) / dx
        m_y[t] = (Phi[1:, :] - Phi[:-1, :]) / dx

    u = pack_u(m_x, m_y, p_int)
    
    rcon = A.dot(u) - b
    rcon_inf = np.linalg.norm(rcon, ord=np.inf)
    if rcon_inf > 1e-10:
        raise RuntimeError(
            f"Initial iterate is infeasible (||A u - b||_inf = {rcon_inf:.3e}). "
            "Check endpoint masses and the Poisson initialization."
        )

    f0, g0, H0 = f_grad_hess(u)
    if not np.isfinite(f0):
        raise RuntimeError("Initialization is not strictly positive.")
    obj_hist = [float(f0)]

    for it in range(max_iter):
        
        fval, grad, H = f_grad_hess(u)
        if not np.isfinite(fval):
            raise RuntimeError("Current iterate is not strictly positive.")

        res = A.dot(u) - b
        rhs = np.concatenate([-grad, -res])
        
        KKT = bmat([[H, A.T],
                    [A, None]], format="csr")
        sol = spsolve(KKT, rhs)
        du = sol[:nvar]
        
        f_prev = fval
        
        alpha = float(alpha0)
        
        p_dir = du[idx_p0:idx_p0 + (T - 1) * N * N]
        alpha_pos = 1.0
        for k in range((T - 1) * N * N):
            if p_dir[k] < 0.0:
                alpha_pos = min(alpha_pos, 0.99 * (u[idx_p0 + k]) / (-p_dir[k]))
        
        alpha = min(alpha, alpha_pos)
        u = u + alpha * du

        print(it, ": alpha: ", alpha, "alpha_pos: ", alpha_pos if alpha_pos != alpha else "")

        f_next, _, _ = f_grad_hess(u)
        if not np.isfinite(f_next):
            raise RuntimeError("Updated iterate is not strictly positive.")
        obj_hist.append(float(f_next))
        den = max(abs(f_prev), 1e-16)
        if abs(f_next - f_prev) / den < tol:
            break


    m_x, m_y, p_int = unpack_u(u)
    p = assemble_p(p_int)
    obj = obj_hist[-1]

    time_grid = np.linspace(0.0, 1.0, p.shape[0])
    
    return m_x, m_y, p, obj, obj_hist, time_grid

def edge_aware(p0, p1, grid_size=16, time_steps=20,
               beta=1e-3, tol=1e-5, max_iter=50, alpha0=0.3,
               eps_weight=0.02, u=None):

    N = int(grid_size)
    L = int(time_steps)
    T = L + 1
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]

    if u.any() == False:
        u = np.zeros((T, N, N))

    eta_x = np.zeros((T, N, N - 1))
    eta_y = np.zeros((T, N - 1, N))
    kappa = np.zeros(T)
    
    for t in range(T-1):
        eta_x[t] = np.abs(u[t, :, :-1] - u[t, :, 1:]) / dx
        eta_y[t] = np.abs(u[t, :-1, :] - u[t, 1:, :]) / dx
    
        vals = np.concatenate([eta_x[t].ravel(), eta_y[t].ravel()])
        nz = vals[vals > 0]
        kappa[t] = np.median(nz) if nz.size else 1.0


    w_x = np.exp(-(eta_x / (kappa[:, None, None] + 1e-16))**2)
    w_y = np.exp(-(eta_y / (kappa[:, None, None] + 1e-16))**2)
    
    w_x_floored = eps_weight + (1 - eps_weight) * w_x
    w_y_floored = eps_weight + (1 - eps_weight) * w_y

    return _newton(p0, p1, grid_size, time_steps, beta, tol, max_iter, alpha0, eps_weight, w_x_floored=w_x_floored, w_y_floored=w_y_floored)