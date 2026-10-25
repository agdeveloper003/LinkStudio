wdm = x.meta.get("wdm") or {}
eta = gn_eta(gamma, att, beta2(d, lam) * 1e27, rs, p.span_km, wdm.get("n_channels", 1),
             wdm.get("spacing_hz")) * a_pre ** 2  # launch power enters the fiber after A_pre
p_ase = 2 * s * rs  # ASE of one span in B = R_s, both polarizations
p_nli = p.n_spans * eta * launch ** 3
self._last.update({"gn_nli_dbm": w_to_dbm(p_nli),
                   "gsnr_gn_db": 10 * np.log10(launch / (p.n_spans * p_ase + p_nli)),
                   "gn_optimum_launch_dbm": w_to_dbm((p_ase / (2 * eta)) ** (1 / 3))})
