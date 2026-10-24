fa = np.abs(f)
if model == "next":
    return 8.536e-15 * n ** 0.6 * fa ** 1.5
length_ft = length_m * 3.281
line_gain = np.abs(line) ** 2 if line is not None else 1.0
return 7.744e-21 * n ** 0.6 * length_ft * fa ** 2 * line_gain
