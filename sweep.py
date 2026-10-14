import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

t0 = time.time()
def checkpoint(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}")

# ============================================================
# CONFIG
# ============================================================
USE_TOY = True   # True: generate + test on toy universe (with ground truth scoring)
                   # False: run on real density field (density_path), no scoring

N = 512
h = 0.678
L = 40.0 / h
dx = L / N
cell_volume = dx**3
smoothing = 2.0

density_path = f"/home/ymtiwari/density_fields/WDM3/d3d_gas_512_z2.400.npy"

merge_threshold = 0.2          # ratio-based merge criterion (Methods 2 & 3)
so_delta_threshold = -0.6      # spherical-underdensity growth threshold (Method 4, Lee et al./Stark et al. style)
so_max_radius_mpc = 8.0        # cap sphere growth -- keep close to toy void radius range for now
so_min_radius_mpc = 0.5

iou_match_threshold = 0.2      # minimum intersection-over-union to call a found void "matched" to ground truth
merge_floor_mode = "deeper"    # merge ratio = ridge / floor; "deeper" -> floor = deeper of the two basin minima (current,
                               # conservative); "shallower" -> floor = shallower basin minimum (classic ZOBOV-like)

# --- merge-threshold sweep (toy mode only) ---
RUN_SWEEP = True
sweep_merge_thresholds = [0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]   # 1.0 = no merging (= "pre")
sweep_iou_thresholds = [0.2, 0.5]       # scored at each; the first one is used for the radius-error panel
sweep_output_prefix = "merge_sweep"     # writes <prefix>.png and <prefix>_<method>.csv

# --- toy model params (only used if USE_TOY) ---
toy_seed = 47
toy_target_filling_fraction = 0.20
toy_radius_range_mpc = (1.0, 6.0)
toy_depth_range = (-1.0, -0.8)
toy_n_halos = 40
toy_halo_depth_range = (3.0, 9.0)
toy_halo_sigma_range_mpc = (0.003, 0.03)
toy_noise_std = 0.0604            # replace with real-data residual estimate when available
toy_merge_threshold_delta = -0.8
toy_radius_to_sigma_k = 1.0        # sigma_cells = radius_cells / k -- k=1 means the ground-truth
                                    # radius sits at 1 sigma of the Gaussian falloff (depth * exp(-0.5) = 0.61*depth,
                                    # i.e. still strongly underdense at the boundary). Try sqrt(2), etc. later.
toy_aniso_log_spread = 0.25        # std of log axis ratios; 0 -> perfectly spherical voids (reproduces the old universe)
toy_aniso_max_ratio = 2.0          # cap on longest/shortest semi-axis
toy_random_orientation = True      # random 3D rotation per void (False -> axes aligned with the grid)

# ============================================================
# SHARED UTILITIES
# ============================================================
def radii_at_level(labels_at_level, n_groups, cell_volume, ndim=3):
    if n_groups == 0:
        return np.array([])
    unique_ids = np.unique(labels_at_level[labels_at_level > 0])
    if len(unique_ids) == 0:
        return np.array([])
    sizes_cells = ndimage.sum(labels_at_level > 0, labels_at_level, unique_ids)
    sizes_physical = sizes_cells * cell_volume
    if ndim == 2:
        return np.sqrt(sizes_physical / np.pi)
    return (3 * sizes_physical / (4 * np.pi))**(1/3)

def sphere_bounding_box(center_ijk, radius_cells, N):
    lo = np.maximum(np.array(center_ijk) - radius_cells, 0).astype(int)
    hi = np.minimum(np.array(center_ijk) + radius_cells + 1, N).astype(int)
    return lo, hi

def random_rotation_matrix(rng):
    A = rng.normal(size=(3, 3))
    Q, Rr = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(Rr))          # make the QR decomposition unique
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1                     # proper rotation, no reflection
    return Q

def draw_triaxial_axes(R, rng, log_spread, max_ratio):
    # three semi-axes whose product is R^3, so the equivalent-sphere radius stays R
    e = rng.normal(0.0, log_spread, size=3)
    e -= e.mean()
    span = e.max() - e.min()
    if span > np.log(max_ratio):
        e *= np.log(max_ratio) / span
    return R * np.exp(e)

def ellipsoid_half_extent(axes, Q):
    # largest half-width of the rotated ellipsoid along any grid axis (for the bounding box)
    return int(np.ceil(np.sqrt(((Q * axes)**2).sum(axis=1)).max()))

def ellipsoid_q(lo, hi, center, axes, Q):
    # q = sum_j (u_j / axes_j)^2, with u = Q^T d the offset in the void's own frame; q <= 1 is inside
    zi, yi, xi = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
                              np.arange(lo[2], hi[2]), indexing='ij')
    d0, d1, d2 = zi - center[0], yi - center[1], xi - center[2]
    q = 0.0
    for j in range(3):
        u = Q[0, j] * d0 + Q[1, j] * d1 + Q[2, j] * d2
        q = q + (u / axes[j])**2
    return q

# ============================================================
# TOY MODEL GENERATOR
# ============================================================
def generate_toy_universe(N, dx, seed=42):
    rng = np.random.default_rng(seed)
    shape_rng = np.random.default_rng(seed + 1)   # separate stream: shapes never shift void positions/depths
    edge_margin_mpc = toy_radius_range_mpc[1] * 1.5
    edge_margin_cells = edge_margin_mpc / dx

    field = np.zeros((N, N, N), dtype=np.float32)
    extent_mask_running = np.zeros((N, N, N), dtype=bool)
    void_params = []

    def add_void_local(field, center_ijk, depth, sigma_axes, Q, extent_sigma=4):
        half = ellipsoid_half_extent(extent_sigma * sigma_axes, Q)
        lo, hi = sphere_bounding_box(center_ijk, half, N)
        q = ellipsoid_q(lo, hi, center_ijk, sigma_axes, Q)
        local_delta = depth * np.exp(-0.5 * q)
        field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.minimum(
            field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], local_delta)

    def stamp_ellipsoid(mask, center_ijk, axes, Q):
        lo, hi = sphere_bounding_box(center_ijk, ellipsoid_half_extent(axes, Q), N)
        q = ellipsoid_q(lo, hi, center_ijk, axes, Q)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= (q <= 1.0)

    checkpoint("  [toy] placing voids")
    n_placed = 0
    while np.mean(extent_mask_running) < toy_target_filling_fraction:
        center_ijk = rng.uniform(edge_margin_cells, N - edge_margin_cells, size=3)
        radius_mpc = rng.uniform(*toy_radius_range_mpc)
        radius_cells = radius_mpc / dx
        depth = rng.uniform(*toy_depth_range)

        axes_cells = draw_triaxial_axes(radius_cells, shape_rng, toy_aniso_log_spread, toy_aniso_max_ratio)
        Q = random_rotation_matrix(shape_rng) if toy_random_orientation else np.eye(3)
        sigma_axes = axes_cells / toy_radius_to_sigma_k

        add_void_local(field, center_ijk, depth, sigma_axes, Q)
        stamp_ellipsoid(extent_mask_running, center_ijk, axes_cells, Q)
        void_params.append({"center": center_ijk, "depth": depth, "radius_cells": radius_cells,
                             "axes_cells": axes_cells, "sigma_axes": sigma_axes, "Q": Q})
        n_placed += 1
    checkpoint(f"  [toy] placed {n_placed} voids, filling fraction = {np.mean(extent_mask_running):.3f}")

    # ---- ground truth: merge grouping via threshold, extent = union of each void's ellipsoid (semi-axes axes_cells, volume-preserving: product = radius_cells^3)
    #      (radius_cells is the primary physical quantity; sigma_cells = radius_cells / k
    #      is derived so the profile is still meaningfully underdense at the boundary) ----
    merge_mask = (field < toy_merge_threshold_delta)
    merge_labels, _ = ndimage.label(merge_mask)

    groups = {}
    next_singleton_id = -1
    for i, v in enumerate(void_params):
        idx = tuple(np.round(v["center"]).astype(int))
        g = merge_labels[idx]
        if g > 0:
            groups.setdefault(g, []).append(i)
        else:
            groups[next_singleton_id] = [i]
            next_singleton_id -= 1

    group_id_grid = np.zeros((N, N, N), dtype=np.int32)
    for new_gid, (g, members) in enumerate(groups.items(), start=1):
        for vi in members:
            v = void_params[vi]
            lo, hi = sphere_bounding_box(v["center"], ellipsoid_half_extent(v["axes_cells"], v["Q"]), N)
            inside = ellipsoid_q(lo, hi, v["center"], v["axes_cells"], v["Q"]) <= 1.0
            group_id_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][inside] = new_gid

    checkpoint(f"  [toy] ground truth: {len(groups)} groups")

    # ---- halos ----
    checkpoint("  [toy] adding halos")
    halo_field = np.zeros((N, N, N), dtype=np.float32)
    for _ in range(toy_n_halos):
        center_ijk = rng.uniform(edge_margin_cells, N - edge_margin_cells, size=3)
        sigma_cells = rng.uniform(*toy_halo_sigma_range_mpc) / dx
        depth = rng.uniform(*toy_halo_depth_range)
        radius_cells = int(np.ceil(5 * sigma_cells))
        lo, hi = sphere_bounding_box(center_ijk, radius_cells, N)
        zi, yi, xi = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]),
                                  np.arange(lo[2], hi[2]), indexing='ij')
        r2 = (zi - center_ijk[0])**2 + (yi - center_ijk[1])**2 + (xi - center_ijk[2])**2
        local_delta = depth * np.exp(-r2 / (2 * sigma_cells**2))
        halo_field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.maximum(
            halo_field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], local_delta)

    field_with_halos = field + halo_field
    noise = rng.normal(0, toy_noise_std, size=(N, N, N)).astype(np.float32)
    final_delta = field_with_halos + noise

    toy_density = (1.0 * (1 + final_delta)).astype(np.float32)
    toy_density = np.clip(toy_density, 1e-4, None)

    return toy_density, group_id_grid

# ============================================================
# TIDAL TENSOR + T-WEB CLASSIFICATION
# ============================================================
def compute_tidal_web(delta_smooth_3d, N, dx):
    checkpoint("computing tidal tensor (FFT)")
    deltak = np.fft.rfftn(delta_smooth_3d)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(N, d=dx), np.fft.fftfreq(N, d=dx),
                              np.fft.rfftfreq(N, d=dx), indexing='ij')
    kmag2 = (kx**2 + ky**2 + kz**2)
    kmag2[0, 0, 0] = 1.0

    T_xx = np.fft.irfftn((kx*kx*deltak)/kmag2).real.astype(np.float32)
    T_xy = np.fft.irfftn((kx*ky*deltak)/kmag2).real.astype(np.float32)
    T_xz = np.fft.irfftn((kx*kz*deltak)/kmag2).real.astype(np.float32)
    T_yy = np.fft.irfftn((ky*ky*deltak)/kmag2).real.astype(np.float32)
    T_yz = np.fft.irfftn((ky*kz*deltak)/kmag2).real.astype(np.float32)
    T_zz = np.fft.irfftn((kz*kz*deltak)/kmag2).real.astype(np.float32)
    checkpoint("streaming eigenvalue classification")

    n = N**3
    web_flat = np.zeros(n, dtype=np.int8)
    chunk = 5_000_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        M = np.zeros((end - start, 3, 3), dtype=np.float32)
        M[:, 0, 0] = T_xx.ravel()[start:end]
        M[:, 1, 1] = T_yy.ravel()[start:end]
        M[:, 2, 2] = T_zz.ravel()[start:end]
        M[:, 0, 1] = M[:, 1, 0] = T_xy.ravel()[start:end]
        M[:, 0, 2] = M[:, 2, 0] = T_xz.ravel()[start:end]
        M[:, 1, 2] = M[:, 2, 1] = T_yz.ravel()[start:end]
        eigs = np.linalg.eigvalsh(M)
        web_flat[start:end] = np.sum(eigs > 0, axis=1)
    del T_xx, T_yy, T_zz, T_xz, T_xy, T_yz
    web_3d = web_flat.reshape(N, N, N).astype(np.int8)
    checkpoint("T-web classification done")
    return web_3d

# ============================================================
# RIDGE-FINDING + MERGE (unchanged logic, bug fixed)
# ============================================================
def find_ridges(labels, field):
    ridges = {}
    max_label = labels.max()

    def process_axis(left, right, left_field, right_field):
        diff_mask = (left != right) & (left > 0) & (right > 0)
        a = left[diff_mask]; b = right[diff_mask]
        hgt = np.maximum(left_field[diff_mask], right_field[diff_mask])
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        key = lo.astype(np.int64) * (max_label + 1) + hi
        order = np.argsort(key)
        key_s, h_s, lo_s, hi_s = key[order], hgt[order], lo[order], hi[order]
        uniq_keys, starts = np.unique(key_s, return_index=True)
        min_h = np.minimum.reduceat(h_s, starts)
        for mh, l, u in zip(min_h, lo_s[starts], hi_s[starts]):
            pair = (int(l), int(u))
            if pair not in ridges or mh < ridges[pair]:
                ridges[pair] = mh

    process_axis(labels[:-1, :, :], labels[1:, :, :], field[:-1, :, :], field[1:, :, :])
    process_axis(labels[:, :-1, :], labels[:, 1:, :], field[:, :-1, :], field[:, 1:, :])
    process_axis(labels[:, :, :-1], labels[:, :, 1:], field[:, :, :-1], field[:, :, 1:])
    return ridges

def prepare_merge(labels, field):
    """Threshold-independent part of the merge: every basin's deepest delta, and the lowest saddle
    (ridge) between every pair of touching basins. Compute once, reuse for any merge threshold."""
    n_basins = int(labels.max())
    if n_basins == 0:
        return 0, {}, {}
    ids = np.arange(1, n_basins + 1)
    mins = ndimage.minimum(field, labels, ids)
    basin_min = dict(zip(ids, mins))
    ridges = find_ridges(labels, field)
    return n_basins, basin_min, ridges

def merge_roots(prepared, merge_threshold, floor_mode=None):
    """Union-find over touching basins: merge a pair when ridge/floor > merge_threshold.
    Returns root_of, an array mapping every basin id (and 0 -> 0) to the id of its merged group."""
    n_basins, basin_min, ridges = prepared
    if floor_mode is None:
        floor_mode = merge_floor_mode
    pick = min if floor_mode == "deeper" else max
    parent = list(range(n_basins + 1))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    for (a, b), ridge_val in ridges.items():
        floor = pick(basin_min[a], basin_min[b])
        if (ridge_val / floor) > merge_threshold:
            union(a, b)
    return np.array([find(i) for i in range(n_basins + 1)])

def merge_weak_basins(labels, field, merge_threshold, prepared=None, floor_mode=None):
    if labels.max() == 0:   # guard: nothing to merge
        return labels.copy(), 0
    if prepared is None:
        prepared = prepare_merge(labels, field)
    root_of = merge_roots(prepared, merge_threshold, floor_mode)
    merged_labels = root_of[labels]
    unique_groups = np.unique(merged_labels[merged_labels > 0])
    lut2 = np.zeros(unique_groups.max() + 1, dtype=np.int64)
    lut2[unique_groups] = np.arange(1, len(unique_groups) + 1)
    final_labels = lut2[merged_labels]
    return final_labels, len(unique_groups)

# ============================================================
# METHOD 4 -- Spherical Underdensity void finder (Lee et al. / Stark et al. style)
# ============================================================
def spherical_underdensity_finder(delta_field, N, dx, seed_min_distance, seed_mask=None,
                                   so_threshold=-0.8, max_radius_mpc=8.0, min_radius_mpc=0.5):
    """
    Grow a sphere from each candidate center (deepest first) until the enclosed
    mean delta rises above so_threshold. Candidates already inside an accepted
    sphere are skipped (standard SO-finder non-overlap rule).
    seed_mask restricts candidate seeds (e.g. to the tidal void mask) -- cuts
    runtime substantially and keeps candidates consistent with Methods 2/3.
    """
    max_radius_cells = max_radius_mpc / dx
    min_radius_cells = min_radius_mpc / dx

    # ---- precompute distance-sorted relative offset kernel ONCE ----
    # The relative geometry (offset -> distance from center) is identical for
    # every seed; only the field values differ. Building this once and reusing
    # it removes a full meshgrid + O(M log M) argsort from every loop iteration.
    r_kernel = int(np.ceil(max_radius_cells))
    offs = np.arange(-r_kernel, r_kernel + 1)
    di_g, dj_g, dk_g = np.meshgrid(offs, offs, offs, indexing='ij')
    dist_g = np.sqrt((di_g**2 + dj_g**2 + dk_g**2).astype(np.float64))
    kernel_order = np.argsort(dist_g.ravel(), kind='stable')
    di_k = di_g.ravel()[kernel_order].astype(np.int32)
    dj_k = dj_g.ravel()[kernel_order].astype(np.int32)
    dk_k = dk_g.ravel()[kernel_order].astype(np.int32)
    dist_k = dist_g.ravel()[kernel_order].astype(np.float32)
    del di_g, dj_g, dk_g, dist_g, kernel_order
    n_kernel = len(dist_k)
    so_chunk_size = 200_000   # shell chunk size for incremental growth

    if seed_mask is not None:
        seeds = peak_local_max(-delta_field, min_distance=seed_min_distance,
                                labels=seed_mask.astype(int))
    else:
        seeds = peak_local_max(-delta_field, min_distance=seed_min_distance)
    seed_vals = delta_field[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    order = np.argsort(seed_vals)          # ascending: most negative (deepest) first
    seeds = seeds[order]

    label_grid = np.zeros((N, N, N), dtype=np.int32)
    accepted_centers = []
    accepted_radii = []
    next_id = 1

    for c in seeds:
        c = c.astype(float)
        skip = False
        for ac, ar in zip(accepted_centers, accepted_radii):
            if np.sum((c - ac)**2) <= ar**2:
                skip = True
                break
        if skip:
            continue

        c_int = np.round(c).astype(np.int64)

        # ---- incremental shell growth: stop as soon as running mean crosses
        # so_threshold instead of always gathering out to max_radius_mpc.
        # Most voids cross well before the cap, so this skips the bulk of the
        # ~2.7M-cell kernel for the common case.
        sum_so_far = 0.0
        count_so_far = 0
        collected_zi, collected_yi, collected_xi, collected_dist = [], [], [], []
        idx_cross = None
        invalid_seed = False
        start = 0
        while start < n_kernel:
            end = min(start + so_chunk_size, n_kernel)
            zi_c = c_int[0] + di_k[start:end]
            yi_c = c_int[1] + dj_k[start:end]
            xi_c = c_int[2] + dk_k[start:end]
            valid_c = (zi_c >= 0) & (zi_c < N) & (yi_c >= 0) & (yi_c < N) & (xi_c >= 0) & (xi_c < N)
            zi_v = zi_c[valid_c]; yi_v = yi_c[valid_c]; xi_v = xi_c[valid_c]
            dist_v = dist_k[start:end][valid_c]
            start = end
            if len(zi_v) == 0:
                continue

            vals = delta_field[zi_v, yi_v, xi_v]
            local_cumsum = np.cumsum(vals, dtype=np.float64) + sum_so_far
            local_counts = np.arange(1, len(vals) + 1) + count_so_far
            running_mean_chunk = local_cumsum / local_counts

            collected_zi.append(zi_v); collected_yi.append(yi_v)
            collected_xi.append(xi_v); collected_dist.append(dist_v)

            above = np.where(running_mean_chunk >= so_threshold)[0]
            if len(above) > 0:
                first = above[0]
                if count_so_far + first == 0:
                    invalid_seed = True   # even the single closest cell alone fails
                else:
                    idx_cross = count_so_far + first - 1
                break
            sum_so_far = local_cumsum[-1]
            count_so_far = local_counts[-1]

        if invalid_seed:
            continue
        if idx_cross is None:
            if count_so_far == 0:
                continue
            idx_cross = int(count_so_far) - 1   # never crosses within max_radius -> cap at max

        all_zi = np.concatenate(collected_zi)
        all_yi = np.concatenate(collected_yi)
        all_xi = np.concatenate(collected_xi)
        all_dist = np.concatenate(collected_dist)

        final_radius = all_dist[idx_cross]
        if final_radius < min_radius_cells:
            continue

        label_grid[all_zi[:idx_cross + 1], all_yi[:idx_cross + 1], all_xi[:idx_cross + 1]] = next_id

        accepted_centers.append(c)
        accepted_radii.append(final_radius)
        next_id += 1

    return label_grid, next_id - 1

# ============================================================
# MATCHING / SCORING vs GROUND TRUTH
# ============================================================
def overlap_table(found_labels, ground_truth_grid):
    """Cells shared by every (found, true) pair that overlaps at all: found ids, true ids, shared cell counts."""
    both = (found_labels > 0) & (ground_truth_grid > 0)
    f = found_labels[both].astype(np.int64)
    t = ground_truth_grid[both].astype(np.int64)
    T = int(ground_truth_grid.max()) + 1
    ukey, counts = np.unique(f * T + t, return_counts=True)
    return ukey // T, ukey % T, counts

def label_sizes(labels):
    """sizes[id] = number of cells carrying that label (sizes[0] is background)."""
    return np.bincount(labels.ravel())

def score_matches(f_ids, t_ids, counts, f_sizes, t_sizes, cell_volume, iou_thresholds):
    """Greedy exclusive one-to-one matching by descending IoU, scored at each IoU threshold.
    f_sizes / t_sizes are arrays indexed by label id (0 = id absent). Returns {thr: (completeness, purity, radius_errors)}."""
    n_found = int(np.count_nonzero(f_sizes[1:]))
    n_true = int(np.count_nonzero(t_sizes[1:]))
    if n_found == 0 or n_true == 0 or len(counts) == 0:
        return {thr: (0.0, 0.0, np.array([])) for thr in iou_thresholds}
    iou = counts / (f_sizes[f_ids] + t_sizes[t_ids] - counts)
    order = np.argsort(-iou, kind="stable")
    f_s, t_s, iou_s = f_ids[order], t_ids[order], iou[order]
    out = {}
    for thr in iou_thresholds:
        n_cand = int(np.searchsorted(-iou_s, -thr, side="left"))      # number of pairs with IoU > thr
        used_f, used_t, mf, mt = set(), set(), [], []
        for fid, tid in zip(f_s[:n_cand].tolist(), t_s[:n_cand].tolist()):
            if fid in used_f or tid in used_t:
                continue
            used_f.add(fid); used_t.add(tid); mf.append(fid); mt.append(tid)
        r_found = (3 * f_sizes[mf] * cell_volume / (4 * np.pi))**(1/3)
        r_true = (3 * t_sizes[mt] * cell_volume / (4 * np.pi))**(1/3)
        out[thr] = (len(mf) / n_true, len(mf) / n_found, (r_found - r_true) / r_true)
    return out

def match_multi(found_labels, ground_truth_grid, cell_volume, iou_thresholds):
    f_ids, t_ids, counts = overlap_table(found_labels, ground_truth_grid)
    return score_matches(f_ids, t_ids, counts, label_sizes(found_labels), label_sizes(ground_truth_grid),
                         cell_volume, iou_thresholds)

def match_to_ground_truth(found_labels, ground_truth_grid, cell_volume, iou_threshold=0.2):
    """
    Returns completeness, purity, and radius errors for matched pairs.
    Matching: greedy exclusive one-to-one assignment by descending IoU; a (found, true)
    pair is accepted only if IoU > threshold and neither member is already matched.
    completeness = matches / n_true, purity = matches / n_found, radius errors per match.
    """
    return match_multi(found_labels, ground_truth_grid, cell_volume, [iou_threshold])[iou_threshold]

def run_merge_sweep(labels_pre, prepared, ground_truth_grid, cell_volume, merge_thresholds, iou_thresholds, floor_mode):
    """Score the merged catalogue at every merge threshold WITHOUT relabelling the full grid: merging basins just
    adds up their cell counts and their overlaps with each true void, so one overlap table is enough."""
    n_pre = prepared[0]
    f_ids0, t_ids0, counts0 = overlap_table(labels_pre, ground_truth_grid)
    pre_sizes = label_sizes(labels_pre)
    t_sizes = label_sizes(ground_truth_grid)
    T = len(t_sizes)
    res = {"thr": np.array(merge_thresholds, dtype=float), "n_voids": [],
           "comp": {i: [] for i in iou_thresholds}, "pur": {i: [] for i in iou_thresholds},
           "rerr": {i: [] for i in iou_thresholds}}
    for thr in merge_thresholds:
        root_of = merge_roots(prepared, thr, floor_mode)
        m_sizes = np.bincount(root_of[1:], weights=pre_sizes[1:n_pre + 1], minlength=n_pre + 1).astype(np.int64)
        ukey, inv = np.unique(root_of[f_ids0] * T + t_ids0, return_inverse=True)
        g_counts = np.bincount(inv.ravel(), weights=counts0).astype(np.int64)
        scores = score_matches(ukey // T, ukey % T, g_counts, m_sizes, t_sizes, cell_volume, iou_thresholds)
        res["n_voids"].append(int(np.count_nonzero(m_sizes[1:])))
        for i in iou_thresholds:
            c, p, r = scores[i]
            res["comp"][i].append(c); res["pur"][i].append(p)
            res["rerr"][i].append(r.mean() if len(r) else np.nan)
    return res

# ============================================================
# MAIN PIPELINE
# ============================================================
if USE_TOY:
    checkpoint("generating toy universe")
    data, ground_truth_grid = generate_toy_universe(N, dx, seed=toy_seed)
    meand = np.mean(data)
    delta = (data - meand) / meand
else:
    checkpoint("loading real density field")
    data = np.load(density_path, mmap_mode="r")
    meand = np.mean(data)
    delta = (data - meand) / meand
    ground_truth_grid = None

checkpoint("smoothing delta field")
delta_smooth_3d = gaussian_filter(delta, sigma=smoothing)

web_3d = compute_tidal_web(delta_smooth_3d, N, dx)
mask_3d = (web_3d == 0)

# ---------------- Method 1 ----------------
checkpoint("Method 1 (ndimage.label)")
labels_cc, n_cc = ndimage.label(mask_3d)

# ---------------- Method 2 ----------------
checkpoint("Method 2 seed search")
coords_m2 = peak_local_max(-delta_smooth_3d, min_distance=int(round(3 * smoothing)),
                            labels=mask_3d.astype(int))
markers_m2 = np.zeros(delta_smooth_3d.shape, dtype=int)
markers_m2[tuple(coords_m2.T)] = np.arange(1, len(coords_m2) + 1)
labels_m2 = watershed(delta_smooth_3d, markers_m2, mask=mask_3d)
n_m2 = labels_m2.max()

# ---------------- Method 3 ----------------
checkpoint("Method 3 seed search")
void_delta_values = delta_smooth_3d[web_3d == 0]
seed_delta_cutoff = np.median(void_delta_values)
basin_delta_cutoff = -0.2
coords_all = peak_local_max(-delta_smooth_3d, min_distance=int(round(3 * smoothing)),
                             labels=mask_3d.astype(int))
seed_vals = delta_smooth_3d[coords_all[:, 0], coords_all[:, 1], coords_all[:, 2]]
coords_m3 = coords_all[seed_vals < seed_delta_cutoff]
markers_m3 = np.zeros(delta_smooth_3d.shape, dtype=int)
markers_m3[tuple(coords_m3.T)] = np.arange(1, len(coords_m3) + 1)
labels_density = watershed(delta_smooth_3d, markers_m3)

ids = np.arange(1, labels_density.max() + 1)
basin_means = ndimage.mean(delta_smooth_3d, labels_density, ids)
bad_ids = ids[basin_means > basin_delta_cutoff]
labels_void_only = labels_density.copy()
labels_void_only[np.isin(labels_density, bad_ids)] = 0
remaining_ids = np.unique(labels_void_only[labels_void_only > 0])
lut = np.zeros(labels_void_only.max() + 1, dtype=np.int64)
lut[remaining_ids] = np.arange(1, len(remaining_ids) + 1)
labels_final = lut[labels_void_only]
n_final = len(remaining_ids)

# ---------------- Method 4 -- spherical underdensity ----------------
checkpoint("Method 4 (spherical underdensity)")
labels_so, n_so = spherical_underdensity_finder(
    delta_smooth_3d, N, dx, seed_min_distance=int(round(3 * smoothing)), seed_mask=mask_3d,
    so_threshold=so_delta_threshold, max_radius_mpc=so_max_radius_mpc, min_radius_mpc=so_min_radius_mpc)
checkpoint(f"Method 4 done -- {n_so} spheres")

# ---------------- merging (Methods 2 & 3) ----------------
checkpoint("preparing merge tables (basin floors + ridges)")
prep_m3 = prepare_merge(labels_final, delta_smooth_3d)
prep_m2 = prepare_merge(labels_m2, delta_smooth_3d)
checkpoint("merging Method 3 basins")
labels_density_merged, nm3 = merge_weak_basins(labels_final, delta_smooth_3d, merge_threshold, prepared=prep_m3)
checkpoint("merging Method 2 basins")
labels_tidal_merged, nm2 = merge_weak_basins(labels_m2, delta_smooth_3d, merge_threshold, prepared=prep_m2)

cases = {
    "M1: ndimage.label":         (labels_cc,             n_cc),
    "M2: tidal+watershed (pre)": (labels_m2,              n_m2),
    "M2: tidal+watershed (mrg)": (labels_tidal_merged,    nm2),
    "M3: raw watershed (pre)":   (labels_final,           n_final),
    "M3: raw watershed (mrg)":   (labels_density_merged,  nm3),
    "M4: spherical underdensity":(labels_so,              n_so),
}

# ============================================================
# STATISTICS
# ============================================================
checkpoint("computing statistics")
radii_dict = {name: radii_at_level(lbl, n, cell_volume, ndim=3) for name, (lbl, n) in cases.items()}

print(f"\n{'Method':<30}{'n_voids':<10}{'mean_r':<10}{'median_r':<10}{'max_r':<10}", end="")
if USE_TOY:
    print(f"{'complete':<10}{'purity':<10}{'r_err_mean':<12}{'r_err_std':<10}")
else:
    print()

if USE_TOY:
    n_gt = len(np.unique(ground_truth_grid[ground_truth_grid > 0]))
    r_gt = radii_at_level(ground_truth_grid, n_gt, cell_volume, ndim=3)
    print(f"{'GROUND TRUTH':<30}{n_gt:<10}{r_gt.mean():<10.3f}{np.median(r_gt):<10.3f}{r_gt.max():<10.3f}"
          f"{'--':<10}{'--':<10}{'--':<12}{'--':<10}")

for name, (lbl, n) in cases.items():
    r = radii_dict[name]
    if len(r) == 0:
        print(f"{name:<30}{'0 (empty)':<10}")
        continue
    line = f"{name:<30}{n:<10}{r.mean():<10.3f}{np.median(r):<10.3f}{r.max():<10.3f}"
    if USE_TOY:
        comp, pur, r_err = match_to_ground_truth(lbl, ground_truth_grid, cell_volume, iou_match_threshold)
        r_err_mean = r_err.mean() if len(r_err) else float('nan')
        r_err_std = r_err.std() if len(r_err) else float('nan')
        line += f"{comp:<10.3f}{pur:<10.3f}{r_err_mean:<12.3f}{r_err_std:<10.3f}"
    print(line)

tidal_void_fraction = np.mean(web_3d == 0)
print(f"\nTidal web void volume fraction: {tidal_void_fraction:.4f}")
if USE_TOY:
    print(f"Ground truth void volume fraction: {np.mean(ground_truth_grid > 0):.4f}")

# ============================================================
# MERGE-THRESHOLD SWEEP (toy mode only)
# ============================================================
if USE_TOY and RUN_SWEEP:
    checkpoint("merge-threshold sweep")
    f1 = lambda c, p: 2 * c * p / (c + p) if (c + p) > 0 else 0.0
    iou_list = list(sweep_iou_thresholds)
    sweep = {}
    for name, labels_pre, prep in [("M2 tidal+watershed", labels_m2, prep_m2),
                                   ("M3 raw watershed", labels_final, prep_m3)]:
        if prep[0] == 0:
            print(f"{name}: no basins, sweep skipped")
            continue
        sweep[name] = run_merge_sweep(labels_pre, prep, ground_truth_grid, cell_volume,
                                      sweep_merge_thresholds, iou_list, merge_floor_mode)
    so_ref = match_multi(labels_so, ground_truth_grid, cell_volume, iou_list)

    print(f"\nMerge sweep (floor mode: {merge_floor_mode}); merge when ridge/floor > threshold, so 1.0 = no merging")
    print(f"M4 reference: " + "  ".join(f"IoU>{i}: comp {so_ref[i][0]:.3f} pur {so_ref[i][1]:.3f} F1 {f1(so_ref[i][0], so_ref[i][1]):.3f}"
                                         for i in iou_list))
    for name, r in sweep.items():
        print(f"\n{name}   (ground truth: {n_gt} voids)")
        head = f"{'merge_thr':<11}{'n_voids':<9}" + "".join(f"c@{i:<6}p@{i:<6}F1@{i:<5}" for i in iou_list) + f"r_err@{iou_list[0]}"
        print(head)
        for k, thr in enumerate(r["thr"]):
            line = f"{thr:<11.2f}{r['n_voids'][k]:<9}"
            for i in iou_list:
                c, pu = r["comp"][i][k], r["pur"][i][k]
                line += f"{c:<8.3f}{pu:<8.3f}{f1(c, pu):<8.3f}"
            line += f"{r['rerr'][iou_list[0]][k]:.3f}"
            print(line)
        f1_first = [f1(c, pu) for c, pu in zip(r["comp"][iou_list[0]], r["pur"][iou_list[0]])]
        kb = int(np.argmax(f1_first))
        print(f"best F1 at IoU>{iou_list[0]}: merge_thr = {r['thr'][kb]:.2f} (F1 = {f1_first[kb]:.3f}, n_voids = {r['n_voids'][kb]})")
        cols = [r["thr"], np.array(r["n_voids"])]
        hdr = "merge_thr,n_voids"
        for i in iou_list:
            cols += [np.array(r["comp"][i]), np.array(r["pur"][i]), np.array(r["rerr"][i])]
            hdr += f",comp_iou{i},pur_iou{i},rerr_iou{i}"
        fname = f"{sweep_output_prefix}_{name.split(':')[0].split()[0]}.csv"
        np.savetxt(fname, np.column_stack(cols), delimiter=",", header=hdr, comments="")

    # ---- plots: one row per method, plt.subplot style ----
    colors_iou = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(20, 4.2 * max(len(sweep), 1)))
    for row, (name, r) in enumerate(sweep.items()):
        base = row * 4
        plt.subplot(len(sweep), 4, base + 1)
        plt.plot(r["thr"], r["n_voids"], 'o-', color='k')
        plt.axhline(n_gt, color='gray', ls='--', label=f"ground truth ({n_gt})")
        plt.yscale('log'); plt.xlabel("merge threshold (1 = no merging)"); plt.ylabel("n_voids")
        plt.title(f"{name}: number of voids", fontsize=10); plt.legend(fontsize=8)

        plt.subplot(len(sweep), 4, base + 2)
        for k, i in enumerate(iou_list):
            ls = '-' if k == 0 else '--'
            plt.plot(r["thr"], r["comp"][i], 'o' + ls, color='tab:blue', label=f"completeness, IoU>{i}")
            plt.plot(r["thr"], r["pur"][i], 's' + ls, color='tab:red', label=f"purity, IoU>{i}")
        plt.ylim(0, 1.02); plt.xlabel("merge threshold"); plt.title(f"{name}: completeness / purity", fontsize=10)
        plt.legend(fontsize=7)

        plt.subplot(len(sweep), 4, base + 3)
        for k, i in enumerate(iou_list):
            ls = '-' if k == 0 else '--'
            plt.plot(r["thr"], [f1(c, pu) for c, pu in zip(r["comp"][i], r["pur"][i])], 'o' + ls,
                     color=colors_iou[k], label=f"F1, IoU>{i}")
            plt.axhline(f1(so_ref[i][0], so_ref[i][1]), color=colors_iou[k], ls=':', alpha=0.7, label=f"M4, IoU>{i}")
        plt.ylim(0, 1.02); plt.xlabel("merge threshold"); plt.title(f"{name}: F1 (dotted = M4)", fontsize=10)
        plt.legend(fontsize=7)

        plt.subplot(len(sweep), 4, base + 4)
        plt.plot(r["thr"], r["rerr"][iou_list[0]], 'o-', color='tab:purple')
        plt.axhline(0, color='gray', ls='--')
        plt.xlabel("merge threshold"); plt.ylabel("mean (r_found - r_true)/r_true")
        plt.title(f"{name}: radius error, IoU>{iou_list[0]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{sweep_output_prefix}.png", dpi=150)
    plt.close()

# ============================================================
# PLOTS
# ============================================================
checkpoint("plotting")
mid = N // 2
slice_cases = {name: lbl[:, :, mid] for name, (lbl, n) in cases.items()}
n_panels = len(slice_cases) + (1 if USE_TOY else 0)

fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
for ax, (name, lbl_slice) in zip(axes, slice_cases.items()):
    ax.imshow(lbl_slice, cmap='nipy_spectral', origin='lower')
    ax.set_title(name, fontsize=9)
if USE_TOY:
    axes[-1].imshow(ground_truth_grid[:, :, mid], cmap='nipy_spectral', origin='lower')
    axes[-1].set_title("Ground truth", fontsize=9)
plt.tight_layout()
plt.savefig('all_methods_comparison_slice.png', dpi=150)
plt.close()

checkpoint("all done")
