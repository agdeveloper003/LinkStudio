def check(self, n_batches: int, elapsed_s: float, metrics: dict[str, float]) -> str | None:
    if n_batches < self.min_batches:
        return None
    if self.metric is not None and self.threshold is not None:
        if self.metric not in metrics:
            raise GraphError(f"Stop metric '{self.metric}' not found. Available: {sorted(metrics)}")
        if metrics[self.metric] >= self.threshold:
            return f"{self.metric} >= {self.threshold:g}"
    if n_batches >= self.max_batches:
        return "max_batches"
    if self.max_time_s is not None and elapsed_s >= self.max_time_s:
        return "max_time"
    return None
