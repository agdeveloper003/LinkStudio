# LinkStudio

Block-based simulator for communication links: wireless, wireline (copper) and optical
physical layers, with a visual editor in the browser.

- **Engine** (`backend/linkstudio`): Python (NumPy/SciPy). Signals flow through a graph of
  blocks (sources, coding, modulation, pulse shaping, OFDM, MIMO, channels, impairments,
  synchronization, equalization, fibre, lasers, photodiodes, copper lines, ...). It supports
  Monte-Carlo BER/BLER sweeps, YAML scenarios and reproducible seeding.
- **User interface** (`frontend/`): React editor for drawing block diagrams, tuning
  parameters, probing signals (constellations, spectra, eye diagrams) and running sweeps.
- **Manual** (`manual/linkstudio-manual.pdf`): theory and user manual. Every system is
  derived first, then traced to its implementation, then validated against simulator output.

> Status: first draft (pre-release). Interfaces may change until v1.0.

## Quick start

Requires Python ≥ 3.11.

```bash
cd backend
pip install -e ".[dev]"
python -m linkstudio serve                              # opens the UI on http://127.0.0.1:8000
python -m linkstudio run scenarios/qpsk_awgn_sweep.yaml # headless run
python -m pytest                                        # test suite
```

## Documentation

| document | content |
|----------|---------|
| [manual/linkstudio-manual.pdf](manual/linkstudio-manual.pdf) | theory and user manual |
| [docs/GUI.md](docs/GUI.md) | user interface guide |
| [docs/ENGINE.md](docs/ENGINE.md) | engine architecture |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | long-term system architecture |
| [backend/README.md](backend/README.md) | scripts, demos and tests |
| [manual/README.md](manual/README.md) | how to rebuild the manual |
