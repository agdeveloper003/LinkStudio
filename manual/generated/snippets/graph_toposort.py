def _toposort(self) -> list[str]:
    succ: dict[str, set[str]] = {n: set() for n in self.blocks}
    for dst, src in self.edges.items():
        if dst.block == src.block:
            raise GraphError(f"Block '{dst.block}' is connected to itself.")
        succ[src.block].add(dst.block)
    indeg = {n: 0 for n in self.blocks}
    for targets in succ.values():
        for t in targets:
            indeg[t] += 1
    ready = [n for n in self.blocks if indeg[n] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for t in sorted(succ[n], key=list(self.blocks).index):
            indeg[t] -= 1
            if indeg[t] == 0:
                ready.append(t)
    if len(order) != len(self.blocks):
        cyclic = [n for n in self.blocks if n not in order]
        raise GraphError(f"The graph contains a cycle involving: {cyclic}")
    return order
