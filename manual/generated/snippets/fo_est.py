nfft = int(2 ** np.ceil(np.log2(4 * n)))
spec = np.abs(np.fft.fft(lp ** m, n=nfft, axis=-1)) ** 2
if spec.ndim == 3:
    spec = spec.sum(axis=1)
f = np.fft.fftfreq(nfft, 1 / fs)
df = f[np.argmax(spec, axis=-1)] / m  # (B,)
self._estimates.extend(df.tolist())
rot = np.exp(-2j * np.pi * df[:, None] * np.arange(n) / fs)
