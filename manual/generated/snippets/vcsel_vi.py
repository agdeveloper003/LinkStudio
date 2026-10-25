i, t = np.asarray(i, dtype=float), np.asarray(t, dtype=float)
vt = K_BOLTZMANN * (t + 273.15) / Q_E
iss = self.b1 * self.iss0 * (t - self.b2)
return self.ideality * vt * np.log(np.maximum(i, 0.0) / iss + 1) + i * self.c1 / (t - self.c2)
