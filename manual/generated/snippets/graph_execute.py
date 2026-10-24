for name in self.order():
    block = self.blocks[name]
    inputs: dict[str, Signal] = {}
    for pname, port in block.input_ports.items():
        ep = Endpoint(name, pname)
        src = self.edges.get(ep)
        if src is not None:
            sig = produced[src]
        elif ep in external:
            sig = external[ep]
        else:
            continue
        if sig.kind not in port.kinds:
            raise BlockError(f"Input '{ep}' accepts {port.kinds.label()} but received {sig.kind.label()}.")
        inputs[pname] = sig
    try:
        outputs = block.process(inputs, ctx)
    except (BlockError, GraphError):
        raise
    except Exception as exc:  # add context, keep the original traceback
        raise BlockError(f"Block '{name}' ({block.type_name}) failed: {exc}") from exc
    for pname in block.output_ports:
        if pname not in outputs:
            raise BlockError(f"Block '{name}' did not produce output '{pname}'.")
        produced[Endpoint(name, pname)] = outputs[pname]
        values[f"{name}.{pname}"] = outputs[pname]
