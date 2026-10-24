shift = np.exp(-2j * np.pi * self.params.channel * spacing * np.arange(n) / fs)
f = np.fft.fftfreq(n, 1 / fs)
if self.params.shape == "rect":
    H = (np.abs(f) <= bw / 2).astype(float)
else:
    H = np.exp(-np.log(2) / 2 * (2 * f / bw) ** 4)  # -3 dB at +-bw/2
y = np.fft.ifft(np.fft.fft(x.data * shift, axis=-1) * H, axis=-1)
