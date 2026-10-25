"""Command line interface.

    python -m linkstudio blocks [--category channel]
    python -m linkstudio describe awgn
    python -m linkstudio run scenarios/qpsk_awgn_sweep.yaml [--csv out.csv]
    python -m linkstudio serve [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import json
import sys

import linkstudio
from linkstudio.core import SimulationResult, block_catalogue, load_scenario


def _cmd_blocks(args: argparse.Namespace) -> int:
    catalogue = block_catalogue()
    by_cat: dict[str, list[str]] = {}
    for name, info in catalogue.items():
        if args.category and info["category"] != args.category:
            continue
        by_cat.setdefault(info["category"], []).append(name)
    for cat in sorted(by_cat):
        print(f"{cat}:")
        for name in by_cat[cat]:
            doc = catalogue[name]["doc"].splitlines()[0] if catalogue[name]["doc"] else ""
            print(f"  {name:<24} {catalogue[name]['label']:<36} {doc}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    info = block_catalogue().get(args.block)
    if info is None:
        print(f"Unknown block '{args.block}'.", file=sys.stderr)
        return 1
    print(f"{info['label']} ({args.block})  [{info['category']} / {info['subcategory']}]\n\n{info['doc']}\n")
    print("Inputs:")
    for p, d in info["inputs"].items():
        print(f"  {p:<8} {d['kinds']}{'' if d['required'] else ' (optional)'}  {d['doc']}")
    print("Outputs:")
    for p, d in info["outputs"].items():
        print(f"  {p:<8} {d['kinds']}  {d['doc']}")
    print("Parameters:")
    for name, schema in info["params"].get("properties", {}).items():
        default = json.dumps(schema.get("default")) if "default" in schema else "required"
        print(f"  {name:<20} default={default:<12} {schema.get('description', '')}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    print(f"Scenario '{scenario.name}'\n")
    print(scenario.graph.describe())
    print()
    result = scenario.run(progress=True)
    print()
    if isinstance(result, SimulationResult):
        print(result.summary())
    else:
        print(result.table())
        if args.csv:
            result.to_csv(args.csv)
            print(f"\nSaved {args.csv}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    import uvicorn

    from linkstudio.api.app import FRONTEND_DIST

    url = f"http://{args.host}:{args.port}"
    if not FRONTEND_DIST.exists():
        print("Frontend not built: run 'npm install && npm run build' in frontend/ (API only for now).")
    print(f"LinkStudio UI on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("linkstudio.api.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linkstudio", description=f"LinkStudio {linkstudio.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("blocks", help="List available blocks.")
    p.add_argument("--category")
    p.set_defaults(func=_cmd_blocks)
    p = sub.add_parser("describe", help="Show ports and parameters of a block.")
    p.add_argument("block")
    p.set_defaults(func=_cmd_describe)
    p = sub.add_parser("run", help="Run a scenario YAML file.")
    p.add_argument("scenario")
    p.add_argument("--csv", help="Save sweep results to CSV.")
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("serve", help="Start the web user interface.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    p.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
