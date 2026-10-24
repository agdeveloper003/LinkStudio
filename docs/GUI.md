# LinkStudio — User interface guide

## Start

```bash
cd backend
python -m linkstudio serve
```

The browser opens on http://127.0.0.1:8000. The UI is already built in
`frontend/dist`; rebuild it after changing the frontend code:

```bash
cd frontend
npm install
npm run build
```

For frontend development with hot reload, run the API with
`python -m linkstudio serve --no-browser` and, in `frontend/`, `npm run dev`
(http://localhost:5173, API calls are proxied to port 8000).

## Layout

| area | what it does |
|------|--------------|
| **Top bar** | scenario browser (click the scenario name or Ctrl+O), New, Save / Save as, undo / redo, automatic layout, Export .py, batch size, seed, validation status |
| **Blocks (left)** | the 86 blocks grouped by category (including wireline and optical); search, drag onto the canvas or double-click to add |
| **Canvas (centre)** | the communication chain. Connect an output (right side of a block) to an input (left side). Handle and edge colors show the signal kind; incompatible connections are refused. Dashed handles are optional inputs. Delete / Backspace removes the selection |
| **Inspector (right)** | rename the selected block, edit its parameters (forms generated from the block schema, sliders for dB values and bounded parameters, `auto` for values taken from upstream metadata) and tick outputs to probe. With nothing selected it shows the scenario name, description and validation errors |
| **Bottom panel** | Live, Simulate and Sweep modes (drag its top edge to resize) |

Signal-kind colors: bits (purple), LLRs (pink), symbols (green), samples (blue),
OFDM grid (amber), CSI (red).

## Editing

* Shift / Ctrl + click, or drag a box, to select several blocks.
* Ctrl+C / Ctrl+V copy and paste blocks together with the connections between them;
  Ctrl+D (or the Duplicate button) duplicates the selection.
* Ctrl+Z / Ctrl+Y undo and redo any change (blocks, connections, parameters, positions, probes).
* Ctrl+L (Layout) arranges the blocks automatically.
* Export .py downloads a standalone script that rebuilds and runs the scenario with the
  Python API, a good starting point for automation.
* The run buttons are disabled while the graph is invalid; the reason is shown next to them.

## Modes

### Live
Runs the chain batch after batch (1–30 Hz) and streams every probe as a scope:
constellation, spectrum, eye diagram, time series, LLR histogram, OFDM grid, channel
response. Parameter changes in the inspector are applied **while running** and the
metrics restart automatically, so moving `awgn.snr_db` or a PA backoff shows the effect
immediately. Structural changes (new blocks or connections) restart the session
automatically.

### Simulate
Monte-Carlo run in the background: a fixed number of batches or "until a metric reaches
a threshold" (e.g. `ber.bit_errors >= 200`). Shows the final metrics (BER with its 95 %
confidence interval) and the probes of the last batch.

### Sweep
Runs the simulation for every value of a parameter (`start:step:stop` or a list), with
an optional second parameter that produces one curve per value. Plots any metric
(log scale for error rates), overlays a theoretical curve (BER vs Eb/N0, so use an AWGN
block in `ebn0` mode) and exports CSV or PNG. *Keep curve* stores the current
curves so that the next runs (other parameters, another scenario) are drawn on top of them.

## Scenario library

38 ready-made scenarios in `backend/scenarios`, grouped by the `category` key:

| category | scenarios |
|----------|-----------|
| Tutorials | BPSK over AWGN, 16-QAM soft demapping, pulse shaping and eye, Rayleigh fading, coding gain (two branches), ISI and adaptive equalization |
| Wireless | single-carrier 16-QAM with impairments, synchronized receiver, OFDM over EVA, OFDM timing/CFO sync, Wi-Fi-like OFDM, LTE-like OFDM, 2x2 MIMO, Alamouti 2x1, satellite 8-PSK with TWTA |
| Coding | LDPC, polar SC, CRC + convolutional over fading, punctured rate 3/4, Hamming |
| Wireline | Fast Ethernet 100BASE-TX (4B/5B, MLT-3, echo cancellation), 10GBASE-KR backplane, 50G PAM-4 SerDes, ADSL DMT, VDSL with FEXT, line-code spectra |
| Optical | OOK receiver sensitivity, O-band APD access link, 56G PAM-4 IM/DD, coherent DP-QPSK long haul, coherent DP-16QAM metro, WDM with fiber nonlinearity, FSO with turbulence |
| Examples (CLI) | the original command-line scenarios (YAML sweep, subsystems, impairments) |

Saving from the UI keeps the category, so your own scenarios appear in the browser too.

## Notes

* Subsystem blocks (e.g. `coded_link_subsystems.yaml`) are shown as single nodes with
  their own ports; their inner graph is edited in its YAML file.
* Saving writes YAML with only non-default parameters plus the block layout and probes,
  so files stay readable and usable from the CLI (`python -m linkstudio run ...`).
