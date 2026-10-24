levels = 1 << p.bits
step = 2 * fs / levels

def q(v: np.ndarray) -> np.ndarray:
    idx = np.clip(np.floor(v / step), -levels // 2, levels // 2 - 1)
    return (idx + 0.5) * step

return {"out": x.evolve(q(x.data.real) + 1j * q(x.data.imag))}
