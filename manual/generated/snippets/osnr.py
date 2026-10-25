def osnr_db(meta: dict, bandwidth: float = REF_BW) -> float | None:
    """OSNR = P / P_ASE with P_ASE = 2 S_ASE B (both polarizations) in the bandwidth B (0.1 nm default)."""
    ase = meta.get("ase_psd")
    power = meta.get("power")
    if not ase or power is None:
        return None
    return 10 * float(np.log10(power / (2 * ase * bandwidth)))
