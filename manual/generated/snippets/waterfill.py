gamma = 10 ** ((gap_db + margin_db) / 10)
b = np.floor(np.log2(1 + snr / gamma)).astype(int)
b = np.clip(b, 0, max_bits)
return np.where(b >= 2, b - (b % 2), b)
