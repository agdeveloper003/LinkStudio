def noise_variance(self, x: Signal) -> float:
    p = self.params
    if p.signal_power is not None:
        power = p.signal_power
    elif p.power_reference == "nominal" and x.meta.get("power") is not None:
        power = float(x.meta["power"])
    else:
        power = float(np.mean(np.abs(x.data) ** 2))
    snr = 10 ** (p.snr_db / 10)
    if p.mode == "snr":
        return power / snr
    sps = x.meta.get("sps", 1)
    if p.mode == "esn0":
        return power * sps / snr
    k = p.bits_per_symbol or x.meta.get("bits_per_symbol")
    if k is None:
        raise BlockError("Eb/N0 mode needs bits_per_symbol (mapper upstream or parameter).")
    rate = p.code_rate or x.meta.get("code_rate", 1.0)
    # Spatial multiplexing carries several symbols per channel use.
    return power * sps / (snr * k * rate * x.meta.get("spatial_rate", 1))
