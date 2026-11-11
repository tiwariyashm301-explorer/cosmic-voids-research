import numpy as np

h = 0.697
Om0, Ob = 0.2814, 0.0464
wX = (Om0 - Ob) * h**2

def T_vogel(k, m):
    alpha = 0.0437 * m**-1.188 * (wX / 0.12)**0.2463 * (h / 0.6736)**2.012   # h^-1 Mpc
    nu = 1.049
    return (1.0 + (alpha * k)**(2 * nu))**(-5.0 / nu)

k, P = np.loadtxt("class_pk_99.dat", unpack=True)

for m in [1.0, 2.0, 3.0, 5.0]:
    Pw = P * T_vogel(k, m)**2
    fname = "class_pk_99_wdm%gkeV.dat" % m
    np.savetxt(fname, np.column_stack([k, Pw]), fmt="%.18e")
    print(fname, " min P =", Pw.min(), " P/P_cdm at k=1,10,40:",
          [float(np.interp(kk, k, Pw / P)) for kk in (1, 10, 40)])                 
