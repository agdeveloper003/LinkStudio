def osnr_db(meta: dict) -> float | None:
    """OSNR in 0.1 nm from the signal power and the accumulated ASE density."""
    ase = meta.get("ase_psd")
    power = meta.get("power")
    if not ase or power is None:
        return None
    return 10 * float(np.log10(power / (2 * ase * REF_BW)))
