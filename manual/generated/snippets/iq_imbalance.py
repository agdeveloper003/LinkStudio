def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
    x = inputs["in"]
    p = self.params
    g = 10 ** (p.amplitude_db / 20)
    phi = np.deg2rad(p.phase_deg)
    k1 = (1 + g * np.exp(-1j * phi)) / 2
    k2 = (1 - g * np.exp(1j * phi)) / 2
    return {"out": x.evolve(k1 * x.data + k2 * np.conj(x.data) + complex(p.dc_i, p.dc_q))}
