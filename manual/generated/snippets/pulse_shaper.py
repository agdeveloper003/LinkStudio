def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
    x = inputs["in"]
    sps = self.params.sps
    y = fir_filter(upsample(x.data, sps), self.taps)
    return {"out": Signal(y, Kind.SAMPLES, fs=x.fs * sps if x.fs else None, meta={
        **x.meta, "sps": sps, "pulse_taps": self.taps, "n_symbols": x.length,
        "power": (x.meta.get("power") or 1.0) / sps,  # unit-energy pulse spreads Es over sps samples
        "delay": x.meta.get("delay", 0.0) + (len(self.taps) - 1) / 2})}
