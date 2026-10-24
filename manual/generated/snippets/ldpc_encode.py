s = (self.H_info @ u.T.astype(np.int32)) % 2  # (m, B)
blocks = s.reshape(self.m // self.z, self.z, -1).astype(np.uint8)
parity = np.bitwise_xor.accumulate(blocks, axis=0).reshape(self.m, -1)
return np.concatenate([u, parity.T.astype(np.uint8)], axis=1)
