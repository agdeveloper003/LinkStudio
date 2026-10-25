d2d = np.asarray(d2d_m, dtype=float)
d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
lf, ld = np.log10(fc_ghz), np.log10(d3d)
if scenario in ("uma", "umi"):
    dbp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / C0
    if scenario == "uma":
        pl1 = 28.0 + 22 * ld + 20 * lf
        pl2 = 28.0 + 40 * ld + 20 * lf - 9 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
        nlos = 13.54 + 39.08 * ld + 20 * lf - 0.6 * (h_ut - 1.5)
        sigma = (4.0, 6.0)
    else:
        pl1 = 32.4 + 21 * ld + 20 * lf
        pl2 = 32.4 + 40 * ld + 20 * lf - 9.5 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
        nlos = 35.3 * ld + 22.4 + 21.3 * lf - 0.3 * (h_ut - 1.5)
        sigma = (4.0, 7.82)
    pl_los = np.where(d2d <= dbp, pl1, pl2)
else:
    pl_los = 32.4 + 17.3 * ld + 20 * lf
    nlos = 38.3 * ld + 17.30 + 24.9 * lf
    sigma = (3.0, 8.03)
pl = np.where(los, pl_los, np.maximum(pl_los, nlos))
return pl, np.where(los, sigma[0], sigma[1])
