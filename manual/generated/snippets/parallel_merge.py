while reason is None and next_merge in ready:
    state, captured = ready.pop(next_merge)
    graph.merge_state(state)
    captures = captured or captures
    next_merge += 1
    reason = done(next_merge, graph.metrics())
