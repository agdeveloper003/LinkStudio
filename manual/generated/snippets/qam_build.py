side = int(round(np.sqrt(order)))
if side * side != order or side & (side - 1):
    raise ValueError(f"Square QAM needs M = 4^n, got {order}.")
half = int(np.log2(side))
levels = _pam_levels(side)
labels = np.arange(order)
points = levels[labels >> half] + 1j * levels[labels & (side - 1)]
return Constellation(points, name=f"{order}-QAM")
