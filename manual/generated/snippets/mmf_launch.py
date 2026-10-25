xi = np.linspace(-12, 12, 4001)  # in units of w0 / sqrt(2) (oscillator length)
dx = xi[1] - xi[0]
s = waist_um / fundamental_um
d = offset_um / fundamental_um * np.sqrt(2)
hermite = np.zeros((n_groups, xi.size))  # normalized oscillator eigenfunctions
hermite[0] = np.pi ** -0.25 * np.exp(-xi ** 2 / 2)
if n_groups > 1:
    hermite[1] = np.sqrt(2) * xi * hermite[0]
for k in range(2, n_groups):
    hermite[k] = np.sqrt(2 / k) * xi * hermite[k - 1] - np.sqrt((k - 1) / k) * hermite[k - 2]

def beam(centre: float) -> np.ndarray:
    g = np.exp(-(xi - centre) ** 2 / (2 * s ** 2))
    return g / np.sqrt(np.sum(g ** 2) * dx)

cx = (hermite @ beam(d) * dx) ** 2
cy = (hermite @ beam(0.0) * dx) ** 2
return np.array([np.sum(cx[:p + 1] * cy[p::-1]) for p in range(n_groups)])
