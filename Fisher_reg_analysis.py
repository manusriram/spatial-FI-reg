import numpy as np
import matplotlib.pyplot as plt
import time
import importlib.util
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib import animation  
from IPython.display import HTML, display
from skimage.metrics import structural_similarity as ssim
import ot
import os
plt.rcParams['text.usetex'] = False

this_dir = Path.cwd()
py_path = this_dir / "Fisher-reg.py"

spec = importlib.util.spec_from_file_location("fisher_reg_module", py_path)
fisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fisher)

_newton = fisher._newton
edge_aware = fisher.edge_aware

def run_fisher(run_id, noise_type, image_type, grid_size, epsilon, weighted):
    N = grid_size
    T = 20
    
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="xy")
    
    def gaussian2d(X, Y, cx, cy, sigma=0.08):
        return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))
    
    def squares():
        p0 = np.zeros((N, N))
        p1 = np.zeros((N, N))
        if N == 20:
            p0[13:19, 1:7] = 1
            p1[1:7, 1:7] = 1
            p1[13:19, 13:19] = 1
        elif N == 16:
            p0[10:15, 1:6] = 1
            p1[1:6, 1:6] = 1
            p1[10:15, 10:15] = 1
        elif N == 24:
            p0[16:23, 1:8] = 1
            p1[1:8, 1:8] = 1
            p1[16:23, 16:23] = 1
        return p0, p1
    
    def mnist():
        try:
            from tensorflow.keras.datasets import mnist as keras_mnist
            (Xtr, ytr), (Xte, yte) = keras_mnist.load_data()
            Xall = np.concatenate([Xtr, Xte], axis=0)
            yall = np.concatenate([ytr, yte], axis=0)
        except Exception:
            from torchvision.datasets import MNIST
            from torchvision import transforms
            ds = MNIST(root="./_mnist", train=True, download=True, transform=transforms.ToTensor())
            Xall = np.stack([ds[i][0].squeeze().numpy() for i in range(len(ds))])
            yall = np.array([ds[i][1] for i in range(len(ds))])
    
        idx4 = np.where(yall == 4)[0][0]
        idx1 = np.where(yall == 1)[0][0]
    
        p0 = Xall[idx4].astype(float)
        p1 = Xall[idx1].astype(float)
    
        for t in range(T):
            p0[t] = np.flipud(p0[t])  
            p1[t] = np.flipud(p1[t])
    
        return p0, p1

    bound = 0.02
    
    if(image_type == "Gaussian"):
        p0 = gaussian2d(X, Y, cx=0.30, cy=0.30, sigma=0.08) + 0.01
        p1 = gaussian2d(X, Y, cx=0.70, cy=0.70, sigma=0.08) + 0.01
    
    else:
        if(image_type == "squares"):
            p0, p1 = squares()
    
        if(image_type == "MNIST"):
            p0, p1 = mnist()
    
        p0 += bound
        p1 += bound
    
    if(noise_type == "Gaussian white"):
        noise_std = 0.02
        p0 = p0 + noise_std * np.random.randn(*p0.shape)
        p1 = p1 + noise_std * np.random.randn(*p1.shape)
    
        p0 = np.clip(p0, 0.0, None)
        p1 = np.clip(p1, 0.0, None)

        p0 += bound
        p1 += bound
    
    p0 /= p0.sum()
    p1 /= p1.sum()

    plot_dir = os.path.join("plots", str(run_id))
    os.makedirs(plot_dir, exist_ok=True)

    plt.imsave(os.path.join(plot_dir, "p0.png"), p0, cmap="gray")
    plt.imsave(os.path.join(plot_dir, "p1.png"), p1, cmap="gray")
    
    print("sum(p0) =", p0.sum(), "sum(p1) =", p1.sum())
    
    def entropic_ot_path(p0, p1, T, reg=1e-2):
        H, W = p0.shape
        n = H * W
    
        a = p0.flatten()
        b = p1.flatten()
        a = a / a.sum()
        b = b / b.sum()
    
        xs = np.linspace(0, 1, H)
        ys = np.linspace(0, 1, W)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    
        M = ot.dist(coords, coords, metric="sqeuclidean")
        M /= M.max()
    
        gamma = ot.sinkhorn(a, b, M, reg)
    
        p_path = []
        for t in np.linspace(0, 1, T):
            T_t = (1 - t) * coords[:, None, :] + t * coords[None, :, :]
            interp = np.zeros((n,))
            for i in range(n):
                interp += gamma[i] * np.exp(
                    -np.sum((coords - ((1 - t) * coords[i] + t * coords))**2, axis=1)
                )
            interp = interp.reshape(H, W)
            interp = np.clip(interp, 0, None)
            interp /= interp.sum()
            p_path.append(interp)
    
        return np.array(p_path)

    u_ref = entropic_ot_path(p0, p1, T=20, reg=1e-2)
    u_ref = np.array([gaussian_filter(u_ref[t], sigma=0.6) for t in range(u_ref.shape[0])])
    u_ref /= u_ref.sum(axis=(1,2), keepdims=True)

    if weighted:
        start_time = time.time()
        
        out = edge_aware(p0, p1, grid_size=N, time_steps=T,
                         beta=1e-3, tol=1e-5, max_iter=50, alpha0=0.3,
                         eps_weight=epsilon, u=u_ref)
        
        end_time = time.time()
    
    else:
        start_time = time.time()
        
        out = _newton(p0, p1, grid_size=N, time_steps=T,
                      beta=1e-3, tol=1e-5, max_iter=50, alpha0=0.3,
                      eps_weight=1.0, w_x_floored=None, w_y_floored=None)
        
        end_time = time.time()
    
    m_x, m_y, p, obj, obj_hist, time_grid = out
    runtime = end_time - start_time
    iterations = len(obj_hist)
    
    fractions = [1/5, 2/5, 3/5, 4/5]
    num_frames = p.shape[0]
    
    ssim_vals = []
    for frac in fractions:
        idx = int(round(frac * (num_frames - 1)))
        val = ssim(p[idx], u_ref[idx], data_range=p.max() - p.min())
        ssim_vals.append(val)
    
    SSIM = np.mean(ssim_vals)
    
    iters = np.arange(len(obj_hist))
    
    plt.figure()
    plt.plot(iters, obj_hist, marker="o")
    plt.xlabel("Newton iteration")
    plt.ylabel("Objective value")
    plt.title("Newton iterations vs objective")
    plt.grid(True)
    
    obj_plot_path = os.path.join(plot_dir, "objective_history.png")
    plt.savefig(obj_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    times = np.linspace(0.0, 1.0, 6)  
    num_frames = p.shape[0]
    idxs = [int(round(t * (num_frames - 1))) for t in times]
    
    plt.figure(figsize=(12, 7))
    
    for k, (t, idx) in enumerate(zip(times, idxs), start=1):
        ax = plt.subplot(2, 3, k)
        ax.imshow(p[idx], origin="lower", cmap="gray")
        ax.set_title(f"t = {t:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    path_plot_path = os.path.join(plot_dir, "transport_snapshots.png")
    plt.savefig(path_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return obj, runtime, SSIM