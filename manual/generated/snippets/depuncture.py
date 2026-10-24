m = llr.shape[-1]
n = x.meta.get("punctured_length")
if n is None:  # shortest mother length whose pattern keeps exactly m positions
    period = len(self.params.pattern)
    big = _keep_mask(self.params.pattern, (m // sum(self.params.pattern) + 1) * period)
    n = int(np.nonzero(np.cumsum(big) == m)[0][0]) + 1
keep = _keep_mask(self.params.pattern, n)
out = np.zeros((*llr.shape[:-1], n))
out[..., keep] = llr
return {"out": Signal(out, Kind.LLR, meta=x.meta)}
