import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, minimum_filter
from skimage.segmentation import watershed

t0 = time.time()
def checkpoint(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ============================================================
# CONFIG
# ============================================================
N = 512
Z = 3.2
L = 40.0                          # h^-1 Mpc, periodic box
dx = L / N
cell_volume = dx**3
box_volume = L**3

smoothing = 2.0                   # cells
min_distance = int(round(3 * smoothing))
merge_threshold = 0.8             # ratio-based
# lambda = 0 threshold is hardcoded in tidal_void_mask

PAD = 96                          # wrap padding for the watershed, in cells
R_MIN = 0.5                       # h^-1 Mpc, reporting cut only
BINS_PER_DEX = 8                  # log-spaced size-function bins per decade
DATA_DIR = "/scratch/ymtiwari/WDM_GRIDS/SHERWOOD"
RUNS = {                          # EDIT: name -> density grid path (same z for all)
    "CDM":  f"{DATA_DIR}/L40N2048/z-{Z}00000/GRIDS/N_GRID-512/d3d_gas_512_z{Z}00.npy",
    "WDM3": f"{DATA_DIR}/L40N2048_WDM3/z-{Z}00000/GRIDS/N_GRID-512/d3d_gas_512_z{Z}00.npy",

}
OUT_DIR = "output"
ROLL_TEST = False                # True: also run on a rolled box to check periodicity
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STAGE 1: smoothing + T-web void mask (periodic, on the 512^3 box only)
# ============================================================
def load_delta(path):
    data = np.load(path, mmap_mode="r")
    mean = data.mean(dtype=np.float64)
    return (data / mean - 1.0).astype(np.float32)

def tidal_void_mask(delta_s):
    dk = np.fft.rfftn(delta_s)
    k1 = np.fft.fftfreq(N, d=dx)
    kr = np.fft.rfftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(k1, k1, kr, indexing='ij', sparse=True)
    ks = [kx, ky, kz]
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    T = [np.fft.irfftn(ks[i] * ks[j] * dk / k2, s=(N, N, N)).astype(np.float32).reshape(-1)
         for i, j in pairs]
    del dk

    n = N**3
    void = np.zeros(n, dtype=bool)
    chunk = 5_000_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        M = np.empty((e - s, 3, 3), dtype=np.float32)
        M[:, 0, 0] = T[0][s:e]; M[:, 1, 1] = T[1][s:e]; M[:, 2, 2] = T[2][s:e]
        M[:, 0, 1] = M[:, 1, 0] = T[3][s:e]
        M[:, 0, 2] = M[:, 2, 0] = T[4][s:e]
        M[:, 1, 2] = M[:, 2, 1] = T[5][s:e]
        void[s:e] = np.linalg.eigvalsh(M)[:, -1] <= 0.0   # all eigenvalues <= 0
    return void.reshape(N, N, N)

# ============================================================
# STAGE 2: seeds, periodic watershed (pad only here), periodic merge
# ============================================================
def find_seeds(delta_s, void):
    field = np.where(void, delta_s, np.inf).astype(np.float32)
    local_min = (field == minimum_filter(field, size=2 * min_distance + 1, mode='wrap')) & void
    markers, n = ndimage.label(local_min, structure=np.ones((3, 3, 3)))
    return markers.astype(np.int32), n

def periodic_watershed(delta_s, markers, void, pad=PAD):
    fp = np.pad(delta_s, pad, mode='wrap')
    mp = np.pad(markers, pad, mode='wrap')
    vp = np.pad(void, pad, mode='wrap')
    lab = watershed(fp, mp, mask=vp)
    del fp, mp, vp
    core = np.ascontiguousarray(lab[pad:-pad, pad:-pad, pad:-pad]).astype(np.int32)
    del lab
    return core

def find_ridges(labels, field):
    """Lowest saddle between each pair of adjacent basins, with wrap-around neighbours."""
    stride = int(labels.max()) + 1
    ridges = {}
    for axis in range(3):
        l2 = np.roll(labels, -1, axis=axis)
        f2 = np.roll(field, -1, axis=axis)
        m = (labels != l2) & (labels > 0) & (l2 > 0)
        a, b = labels[m], l2[m]
        h = np.maximum(field[m], f2[m])
        key = np.minimum(a, b).astype(np.int64) * stride + np.maximum(a, b)
        order = np.argsort(key)
        ks, hs = key[order], h[order]
        uk, start = np.unique(ks, return_index=True)
        mh = np.minimum.reduceat(hs, start)
        for k, v in zip(uk.tolist(), mh.tolist()):
            if k not in ridges or v < ridges[k]:
                ridges[k] = v
    return ridges, stride

def merge_basins(labels, field, thr):
    n = int(labels.max())
    floor = np.concatenate([[0.0], ndimage.minimum(field, labels, np.arange(1, n + 1))])
    ridges, stride = find_ridges(labels, field)
    parent = np.arange(n + 1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for key, ridge in ridges.items():
        a, b = divmod(key, stride)
        fl = min(floor[a], floor[b])
        if fl < 0 and ridge / fl > thr:        # near 1 = weak wall
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
    roots = np.array([find(i) for i in range(n + 1)])
    uniq = np.unique(roots[1:])
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[uniq] = np.arange(1, len(uniq) + 1)
    return lut[roots[labels]], len(uniq)

def basin_sizes(labels, n):
    return np.bincount(labels.ravel(), minlength=n + 1)[1:]

def radii_from_sizes(sizes):
    return (3 * sizes * cell_volume / (4 * np.pi))**(1 / 3)

def run_m2(path, shift=None):
    checkpoint(f"loading {path}")
    delta = load_delta(path)
    if shift is not None:
        delta = np.roll(delta, shift, axis=(0, 1, 2))
    delta_s = gaussian_filter(delta, sigma=smoothing, mode='wrap')
    del delta
    checkpoint("tidal void mask (512^3)")
    void = tidal_void_mask(delta_s)
    checkpoint("seeds")
    markers, n_seeds = find_seeds(delta_s, void)
    checkpoint(f"{n_seeds} seeds, periodic watershed")
    lab_pre = periodic_watershed(delta_s, markers, void)
    del markers
    checkpoint("merging")
    lab_mrg, n_mrg = merge_basins(lab_pre, delta_s, merge_threshold)
    s_pre = basin_sizes(lab_pre, n_seeds)
    s_mrg = basin_sizes(lab_mrg, n_mrg)
    return dict(void=void, lab_pre=lab_pre, n_pre=n_seeds, lab_mrg=lab_mrg, n_mrg=n_mrg,
                r_pre=radii_from_sizes(s_pre), r_mrg=radii_from_sizes(s_mrg),
                fill=void.mean(), fill_lab=(lab_mrg > 0).mean(),
                perc_pre=s_pre.max() / s_pre.sum(),     # share of void volume in largest void
                perc_mrg=s_mrg.max() / s_mrg.sum())

# ============================================================
# run all fields at this redshift
# ============================================================
results = {}
for name, path in RUNS.items():
    results[name] = run_m2(path)
    res = results[name]
    np.save(f"{OUT_DIR}/labels_merged_{name}_z{Z}.npy", res["lab_mrg"])
    np.save(f"{OUT_DIR}/void_mask_{name}_z{Z}.npy", res["void"])

# ============================================================
# STAGE 3: counts, filling fraction, percolation check, size function
# ============================================================
print(f"\n{'run':<8}{'fill':<9}{'fill_lab':<10}{'N_pre':<9}{'N_mrg':<9}{'N(R>Rmin)':<11}"
      f"{'mean_R':<9}{'median_R':<10}{'max_R':<9}{'big_share_pre':<15}{'big_share_mrg':<15}")
for name, res in results.items():
    r = res["r_mrg"]
    print(f"{name:<8}{res['fill']:<9.4f}{res['fill_lab']:<10.4f}{res['n_pre']:<9}{res['n_mrg']:<9}"
          f"{(r > R_MIN).sum():<11}{r.mean():<9.3f}{np.median(r):<10.3f}{r.max():<9.3f}"
          f"{res['perc_pre']:<15.4f}{res['perc_mrg']:<15.4f}")

# shared bin edges, no upper cut: run from R_MIN up to the largest void in any run
r_top = max(res["r_mrg"].max() for res in results.values())
nb = max(1, int(np.ceil(np.log10(r_top / R_MIN) * BINS_PER_DEX)))
edges = R_MIN * 10**(np.arange(nb + 1) / BINS_PER_DEX)
r_cen = np.sqrt(edges[:-1] * edges[1:])
dlnr = np.diff(np.log(edges))

def size_function(r):
    cnt, _ = np.histogram(r, bins=edges)
    norm = box_volume * dlnr
    return cnt / norm, np.sqrt(cnt) / norm, cnt     # dn/dlnR [h^3 Mpc^-3], Poisson error, counts

# ============================================================
# STAGE 4: plots
# ============================================================
names = list(results)
ref = names[0]

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
for name, res in results.items():
    y, err, _ = size_function(res["r_mrg"])
    plt.errorbar(r_cen, y, yerr=err, marker='o', capsize=2, label=f"{name} merged")
    y0, _, _ = size_function(res["r_pre"])
    plt.plot(r_cen, y0, ls='--', alpha=0.5, label=f"{name} pre-merge")
plt.xscale('log'); plt.yscale('log')
plt.xlabel('R  [h$^{-1}$ Mpc]'); plt.ylabel('dn/dlnR  [h$^3$ Mpc$^{-3}$]')
plt.title('M2 void size function'); plt.legend(fontsize=8)
plt.subplot(1, 3, 2)
for name, res in results.items():
    r = np.sort(res["r_mrg"])
    r = r[r > R_MIN]
    plt.plot(r, (len(r) - np.arange(len(r))) / box_volume, label=name)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('R  [h$^{-1}$ Mpc]'); plt.ylabel('n(>R)  [h$^3$ Mpc$^{-3}$]')
plt.title('cumulative'); plt.legend()
plt.subplot(1, 3, 3)
y_ref, e_ref, c_ref = size_function(results[ref]["r_mrg"])
for name in names[1:]:
    y, e, c = size_function(results[name]["r_mrg"])
    ok = (c > 0) & (c_ref > 0)
    ratio = y[ok] / y_ref[ok]
    rerr = ratio * np.sqrt(1 / c[ok] + 1 / c_ref[ok])
    plt.errorbar(r_cen[ok], ratio, yerr=rerr, marker='o', capsize=2, label=f"{name}/{ref}")
plt.axhline(1, color='k', lw=0.8)
plt.xscale('log')
plt.xlabel('R  [h$^{-1}$ Mpc]'); plt.ylabel('size function ratio')
plt.title('merged, relative to ' + ref); plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m2_size_function_z{Z}.png", dpi=150)
plt.close()
mid = N // 2
plt.figure(figsize=(6 * len(results), 5))
for i, (name, res) in enumerate(results.items()):
    plt.subplot(1, len(results), i + 1)
    plt.imshow(res["lab_mrg"][:, :, mid], cmap='nipy_spectral', origin='lower')
    plt.title(f"{name}: merged voids, mid-slice\nfill={res['fill']:.3f}, N={res['n_mrg']}", fontsize=9)
    plt.xlabel('grid cells')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m2_slices_z{Z}.png", dpi=150)
plt.close()
# ============================================================
# periodicity check: same box, rolled by an arbitrary offset
# ============================================================
if ROLL_TEST:
    name = names[0]
    rolled = run_m2(RUNS[name], shift=(N // 3, N // 5, N // 7))
    r0 = results[name]
    print(f"\nroll test ({name}):  fill {r0['fill']:.5f} vs {rolled['fill']:.5f}   "
          f"N_pre {r0['n_pre']} vs {rolled['n_pre']}   N_mrg {r0['n_mrg']} vs {rolled['n_mrg']}")
checkpoint("all done")
