def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
    x = inputs["in"]
    fs = _require_fs(x, "Phase noise")
    var = 2 * np.pi * self.params.linewidth_hz / fs
    phi = np.cumsum(self.rng.standard_normal(x.data.shape) * np.sqrt(var), axis=-1)
    return {"out": x.evolve(x.data * np.exp(1j * phi))}
