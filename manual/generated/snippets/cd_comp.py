H = np.conj(dispersion_operator(x.length, fs, cd, lam, slope))
y = np.fft.ifft(np.fft.fft(x.data, axis=-1) * H, axis=-1)
