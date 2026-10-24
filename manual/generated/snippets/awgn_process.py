def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
    x = inputs["in"]
    n0 = self.noise_variance(x)
    if np.iscomplexobj(x.data):
        noise = self.rng.standard_normal((*x.data.shape, 2)).view(np.complex128)[..., 0]
    else:  # real baseband (wireline, IM/DD): N0/2 per real dimension
        noise = self.rng.standard_normal(x.data.shape)
    y = x.data + np.sqrt(n0 / 2) * noise
    return {"out": x.evolve(y, noise_var=x.meta.get("noise_var", 0.0) + n0)}
