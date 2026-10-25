p = self.params
h = np.exp(-2 * (np.pi * f * sigma_cd_s) ** 2)
if self.groups is None:
    return h * np.exp(-2 * (np.pi * f * self.modal_rms_s()) ** 2)
w, d = self.groups
modal = np.exp(-2j * np.pi * np.multiply.outer(f, d * p.length_m)) @ (w / w.sum())
return h * modal
