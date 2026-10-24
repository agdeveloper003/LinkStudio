r = poly.bit_length() - 1
rems = np.empty(k, dtype=np.int64)
rem = poly ^ (1 << r)  # x^r mod g
for j in range(k):
    rems[k - 1 - j] = rem
    rem <<= 1
    if rem >> r & 1:
        rem ^= poly
shifts = np.arange(r - 1, -1, -1)
return ((rems[:, None] >> shifts) & 1).astype(np.int32)
