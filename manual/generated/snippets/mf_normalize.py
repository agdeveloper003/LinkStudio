y = fir_filter(x.data, rx_taps)
n_sym = x.meta.get("n_symbols", x.length // sps)
start = int(round(x.meta.get("delay", 0.0) + (len(rx_taps) - 1) / 2))

noise_gain = float(np.sum(np.abs(rx_taps) ** 2))
if p.normalize and tx_taps is not None:
    overall = np.convolve(tx_taps, rx_taps)
    peak = overall[int(round((len(tx_taps) - 1) / 2 + (len(rx_taps) - 1) / 2))]
    y = y / peak
    noise_gain /= abs(peak) ** 2
noise = {"noise_var": x.meta["noise_var"] * noise_gain} if x.meta.get("noise_var") is not None else {}
