for i, point in enumerate(points):
    if should_stop is not None and should_stop():
        break
    for key, value in point.items():
        self.graph.set_param(key, value)
    sim = Simulation(self.graph, self.batch_size, self.seed if self.same_seed else self.seed + i)
    res = sim.run(n_batches=self.n_batches, stop=self.stop, on_batch=keep_going)
