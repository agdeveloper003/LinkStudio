hp = grid.data[..., lay.bins(lay.pilot_freqs)] / lay.pilot_values
if self.params.average_symbols:
    hp = np.broadcast_to(hp.mean(axis=1, keepdims=True), hp.shape)
# Remove the linear phase of the dominant delay (timing offset inside the CP)
# before interpolating, then restore it.
pf = lay.pilot_freqs
gaps = np.diff(pf)
if gaps.size:
    step = np.bincount(gaps).argmax()
    pairs = np.nonzero(gaps == step)[0]
    slope = np.angle(np.sum(hp[..., pairs + 1] * np.conj(hp[..., pairs]), axis=-1)) / step
else:
    slope = np.zeros(hp.shape[:-1])
hp = hp * np.exp(-1j * slope[..., None] * pf)
interp = self._interp_matrix(lay)
H = np.ones((B, n_sym, n_fft), dtype=np.complex128)
H[..., lay.bins(lay.active_freqs)] = (hp @ interp.T) * np.exp(1j * slope[..., None] * lay.active_freqs)
