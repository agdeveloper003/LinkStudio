def reset(self, seed: int, path: str) -> None:
    """Re-seed the block random generator and clear runtime state.

    The generator is derived from the master ``seed`` and the block ``path``
    (its hierarchical name), so every block gets an independent but
    reproducible random stream.
    """
    self.rng = np.random.default_rng([seed, zlib.crc32(path.encode())])
    self.__dict__["_recording"] = set()
    try:
        self.on_reset()
    finally:
        self.__dict__["_state_keys"] = tuple(sorted(self.__dict__.pop("_recording")))
