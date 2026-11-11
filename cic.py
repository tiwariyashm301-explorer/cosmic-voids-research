import numpy as np
import bigfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def cic_density(pos, boxsize, ngrid):
    """3D CIC overdensity delta on an ngrid^3 periodic grid. pos and boxsize in the same units."""
    x = pos / boxsize * ngrid                 # positions in grid-cell units
    i0 = np.floor(x).astype(np.int64)         # lower-left cell of each particle
    d = x - i0                                # fractional offset inside the cell (0..1)
    rho = np.zeros(ngrid**3)
    for dx in (0, 1):
        wx = d[:, 0] if dx else 1 - d[:, 0]
        ix = (i0[:, 0] + dx) % ngrid          # % wraps around the periodic box
        for dy in (0, 1):
            wy = d[:, 1] if dy else 1 - d[:, 1]
            iy = (i0[:, 1] + dy) % ngrid
            for dz in (0, 1):
                wz = d[:, 2] if dz else 1 - d[:, 2]
                iz = (i0[:, 2] + dz) % ngrid
                idx = (ix * ngrid + iy) * ngrid + iz
                rho += np.bincount(idx, weights=wx * wy * wz, minlength=ngrid**3)
    rho = rho.reshape(ngrid, ngrid, ngrid)
    return rho / rho.mean() - 1.0             # overdensity delta

def slab_map(snapdir, ngrid=256, zlo=0.0, zhi=5.0):
    bf = bigfile.BigFile(snapdir)
    L = bf["Header"].attrs["BoxSize"][0] / 1000.0          # Mpc/h
    pos = bf["1/Position"][:].astype(np.float64) / 1000.0  # Mpc/h
    delta = cic_density(pos, L, ngrid)
    k0, k1 = int(zlo / L * ngrid), int(zhi / L * ngrid)
    return delta[:, :, k0:k1].mean(axis=2), L              # projected slab, shape (ngrid, ngrid)

cdm_dir = "cdm_512_40mpc/output/PART_003"
wdm_dir = "wdm_512_40mpc_1keV/output/PART_003"

dc, L = slab_map(cdm_dir, ngrid=256, zlo=0.0, zhi=0.2)
dw, _ = slab_map(wdm_dir, ngrid=256, zlo=0.0, zhi=0.2)
res = dw - dc

floor = 1e-3   # caps the log at -2 for completely empty regions
lc = np.log10(np.maximum(1 + dc, floor))
lw = np.log10(np.maximum(1 + dw, floor))

# one shared scale for the two density panels, from the pooled data
both = np.concatenate([lc.ravel(), lw.ravel()])
vmin, vmax = np.percentile(both, [1, 99.5])

# symmetric scale for the residual, so white means zero difference
rm = np.percentile(np.abs(res), 99.5)

print("density scale:", vmin, vmax, " residual scale: +/-", rm)
print("rms(res)/rms(dc) =", res.std() / dc.std())

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.imshow(lc.T, origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax)
plt.colorbar(fraction=0.046, pad=0.04, label="log10(1 + delta)")
plt.title("CDM")
plt.xlabel("x [Mpc/h]")
plt.ylabel("y [Mpc/h]")

plt.subplot(1, 3, 2)
plt.imshow(lw.T, origin="lower", extent=[0, L, 0, L], vmin=vmin, vmax=vmax)
plt.colorbar(fraction=0.046, pad=0.04, label="log10(1 + delta)")
plt.title("MWDM_therm = 1 keV")
plt.xlabel("x [Mpc/h]")

plt.subplot(1, 3, 3)
plt.imshow(res.T, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r", vmin=-rm, vmax=rm)
plt.colorbar(fraction=0.046, pad=0.04, label="delta_WDM - delta_CDM")
plt.title("Residual")
plt.xlabel("x [Mpc/h]")

plt.tight_layout()
plt.savefig("cic_cdm_wdm_z2_40mpc_512.png", dpi=150)
bf_cdm = bigfile.BigFile(cdm_dir)
bf_wdm = bigfile.BigFile(wdm_dir)

print("CDM BoxSize:", bf_cdm["Header"].attrs["BoxSize"])
print("WDM BoxSize:", bf_wdm["Header"].attrs["BoxSize"])

pos_cdm = bf_cdm["1/Position"][:].astype(np.float64) / 1000.0
pos_wdm = bf_wdm["1/Position"][:].astype(np.float64) / 1000.0

print("CDM Position shape:", pos_cdm.shape)
print("WDM Position shape:", pos_wdm.shape)

print("CDM Position range:", pos_cdm.min(), pos_cdm.max())
print("WDM Position range:", pos_wdm.min(), pos_wdm.max())
