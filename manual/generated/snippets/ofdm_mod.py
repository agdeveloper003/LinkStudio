n_sym = -(-n // lay.n_data)
data = np.zeros((B, n_sym * lay.n_data), dtype=np.complex128)
data[:, :n] = x.data
grid = np.zeros((B, n_sym, lay.n_fft), dtype=np.complex128)
grid[..., lay.bins(lay.data_freqs)] = data.reshape(B, n_sym, lay.n_data)
grid[..., lay.bins(lay.pilot_freqs)] = lay.pilot_values
time = np.fft.ifft(grid, axis=-1, norm="ortho")
with_cp = np.concatenate([time[..., lay.n_fft - lay.cp_len:], time], axis=-1)
fs = self.params.subcarrier_spacing * lay.n_fft
data_power = (x.meta.get("power") or 1.0)
n_pilots = lay.pilot_freqs.size
meta = {**x.meta, "ofdm": lay, "n_data_symbols": n, "sps": lay.n_fft / lay.active_freqs.size,
        "delay": 0.0, "data_power": data_power,
        "power": (lay.n_data * data_power + n_pilots) / lay.n_fft}
frame = with_cp.reshape(B, -1)
