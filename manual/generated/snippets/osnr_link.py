a = 10 ** (-span_loss_db / 10)
g = 1 / a if gain_db is None else 10 ** (gain_db / 10)
ag = a * g
acc = n_spans if abs(ag - 1) < 1e-12 else (1 - ag ** n_spans) / (1 - ag)
p_rx = 1e-3 * 10 ** (p_tx_dbm / 10) * ag ** n_spans
return float(10 * np.log10(p_rx / (ase_power(g, nf_db, bandwidth, wavelength_nm) * acc)))
