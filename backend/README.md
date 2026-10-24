# LinkStudio backend — PHY simulation engine and API

Modular, block-based communication system simulator in Python (NumPy/SciPy) with a
FastAPI server for the React user interface.
Architecture: [../docs/ENGINE.md](../docs/ENGINE.md) · UI guide: [../docs/GUI.md](../docs/GUI.md).

## Requirements

Python ≥ 3.11 with `numpy`, `scipy`, `pydantic`, `pyyaml`, `matplotlib` (plots),
`fastapi` + `uvicorn` (UI) and `pytest` (tests). Optional editable install:

```bash
pip install -e ".[dev]"
```

The scripts and tests also work without installing (run them from `backend/`).

## User interface

```bash
python -m linkstudio serve
```

## Scripts

```bash
python scripts/main.py            # guided tour + 34 theory checks (~100 s)
python scripts/main.py --quick    # faster
python scripts/main.py --only wireline optical --show
```

Single topics: `demo_modulation.py`, `demo_pulse_shaping.py`, `demo_channels.py`,
`demo_coding.py`, `demo_ofdm.py`, `demo_sync.py`, `demo_mimo.py`, `demo_wireline.py`,
`demo_optical.py`, `demo_scenarios.py`.
Figures and CSV files are written to `backend/output/`.

## Tests

```bash
python -m pytest
```

## Command line

```bash
python -m linkstudio blocks
python -m linkstudio describe ldpc_decoder
python -m linkstudio run scenarios/qpsk_awgn_sweep.yaml --csv output/qpsk.csv
python -m linkstudio serve --port 8000
```

## Minimal example

```python
import linkstudio as cs

g = cs.Graph("qpsk")
g.create("src", "random_bits", n_bits=2000)
g.create("map", "mapper", scheme="psk", order=4)
g.create("awgn", "awgn", mode="ebn0", snr_db=6.0)
g.create("demap", "demapper", method="hard")
g.create("ber", "ber_meter")
g.chain("src", "map", "awgn", "demap", "ber.rx")
g.connect("src", "ber.tx")

result = cs.Simulation(g, batch_size=64, seed=1).run(
    stop=cs.StopCriteria.errors("ber.bit_errors", 200))
print(result.summary())
```
