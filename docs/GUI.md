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
| **Top bar** | scenario browser (click the scenario name or Ctrl+O), New, Save / Save as, undo / redo, automatic layout, keyboard shortcuts (?), Export .py, seed, validation status |
| **Blocks (left)** | the 94 blocks grouped by functional class (SOURCES, ENCODERS, MODULATORS, TRANSMISSION MEDIA, DECODERS...), the larger ones split into subclasses (code families, media...), shown with readable names; complementary classes (encoders / decoders, mappers / demappers...) share a color. Search matches names, identifiers and docs; drag onto the canvas or double-click to add. The switch next to the search toggles the tree view (names) and the grid view: every block drawn exactly as on the canvas, scaled by the Zoom slider, as many per row as the panel width allows |
| **Canvas (centre)** | the communication chain. Connect an output (right side of a block by default) to an input (left side). Handle and edge colors show the signal kind; incompatible connections are refused. Dashed handles are optional inputs. Links are routed with orthogonal segments around the blocks, never behind them. Delete / Backspace removes the selection; parallel links from different outputs sharing a channel are drawn side by side |
| **Inspector (right)** | rename the selected block, edit its parameters (forms generated from the block schema, sliders for dB values and bounded parameters, `auto` for values taken from upstream metadata). Linear blocks (cables, CTLE, lasers and VCSELs, photodiodes, fibers, pulse shapers) also show their frequency response (magnitude, phase or group delay); random multipath channels show an estimate of one realization. With nothing selected it shows the scenario name, description and validation errors |
| **Scope** | the Scope (MEASUREMENTS class) is an instrument with a screen. The plot selector under its name lists the views that suit the displayed signal: time, spectrum, constellation and eye diagram for waveforms; constellation, time and spectrum for symbols; LLR histogram for LLRs; constellation and resource grid for OFDM grids; channel response for CSI. Auto picks the natural one (spectrum for waveforms, constellation for symbols...). After a Simulate run every view is already computed, so switching needs no new run; in Live the next frame shows it, without restarting the measurements. It has up to four inputs: the inputs enter from the bottom side; drag a free input onto a link (or drop the scope from the library onto a link) to tap the output feeding it, drawn as a solid junction dot just after the port. The selector in the top-right corner shows the displayed signal and switches between the tapped ones; drag the bottom-right corner to resize the scope (the size is saved with the scenario) |
| **Display blocks** | every block shows its function. The signal-processing blocks draw it under their header in the colour of their class: codewords with parity (block codes, CRC, 64b/66b), the shift register of the convolutional encoder, the Viterbi trellis, the LDPC Tanner graph, the polar butterfly, puncturing patterns, the interleaver (the permutation drawn from its seed spreading a burst of adjacent bits apart, or the block matrix numbered in write order), the scrambler (its LFSR with the real taps and seed, and a run of zeros coming out scrambled), line-code waveforms, the pulse shape, the subcarrier comb (OFDM) or bit loading (DMT), FIR taps, the quantizer staircase, tapped delay lines (LMS, DFE, linear and echo equalizers), the CTLE response, PLLs, correlation peaks, the EVM meter with live EVM and MER; the physical ones are drawings without a frame: the AWGN adder with its noise source, mixers with their oscillator (frequency offset, phase noise), the PA with its AM/AM curve, up / down samplers, twisted pairs (copper line, crosstalk), antennas with fading or multipath rays, the path-loss mast and terminal, FSO telescopes, WDM multiplexers, Mach-Zehnder and IQ modulators, the coherent receiver and the electrical power meter (an analog gauge fed from the bottom). Some blocks show more: Random Bits shows its bit stream (scrolling while Live runs) with the transition probability (share of bits that differ from the previous one) set by a slider or typed in, or `i.i.d.` for independent bits; the PRBS Generator draws its shift register with the XOR feedback; the BER Meter has a status LED (steady green, red whenever new errors arrive) with live BER, bits and errors; the Constellation Mapper draws its constellation (resizable); the Optical Power Meter is a handheld meter with the live power in dBm, fed from its bottom connector. The lasers are drawings too: the DML a butterfly package with its fiber pigtail, the VCSEL its cross-section (DBR mirrors, active layer, oxide aperture) with the beam through a coupling lens, coloured by the wavelength. The link media are drawings without a block frame: the MIMO Channel is two mirrored arrays of `-<|` antennas with the rays between them (live strength |h|), the fibers and the amplified link a line with a winding resting on it (and amplifiers), with a light pulse while Live runs; hover or select them to see their name |
| **Side panels** | Blocks and Inspector can be closed (× or Ctrl+B / Ctrl+I, reopen from the top-bar toggles) and resized by dragging their inner edge; the layout is remembered |
| **Run bar (top, after seed)** | Live / Simulate / Sweep switch, Run / Stop, ⚙ settings of the selected mode (rate, stop rule, frames/batch, cores, sweep parameters, theory) and a results toggle. Metrics and sweep curves open in a floating window that can be moved, resized and closed |

Signal-kind colors: bits (purple), LLRs (pink), symbols (green), samples (blue),
OFDM grid (amber), CSI (red).

## Editing

Press **?** (or the ? button in the top bar) for the list of keyboard shortcuts.

* Selection: click a block; Shift / Ctrl + click adds blocks; Shift + drag on the empty
  canvas selects the blocks in a box; Ctrl+A selects all, Esc clears the selection.
* With several blocks selected a bar over the canvas offers: **Align row** (the blocks
  line up with the leftmost one so that the links between them are straight), **Align
  column**, **Distribute** (equal gaps), **Flip**, **Duplicate** and **Delete**.
* Ctrl+C / Ctrl+X / Ctrl+V copy, cut and paste blocks together with the connections
  between them; paste puts the copy at the mouse (pasting again steps the copies).
  Ctrl+D duplicates, Delete / Backspace deletes, the arrow keys move the selection
  (Shift + arrow: further).
* Ctrl+Z / Ctrl+Y undo and redo any change (blocks, connections, parameters, positions,
  port places, probes).
* **Direction**: a block faces right (inputs on the left, outputs on the right) or left.
  F (or Flip in the inspector / selection bar) turns the selected blocks around. A block
  dragged to the other side of the blocks it is linked to turns around by itself, so a
  chain can run right to left, or turn back on itself, without moving any port; the
  drawings of the blocks follow the direction (their text stays readable; plots against
  frequency or time keep their axes).
* Ctrl + drag a port to move it anywhere on the block border (any side), e.g. for feedback
  paths; the places are saved in the scenario.
* While a block is dragged it snaps so that its connected ports line up with the other end
  (straight links), or so that its edges / centre line up with a neighbouring block; a yellow
  guide shows the alignment.
* **Layout** (Ctrl+L) arranges the blocks along the signal flow: blocks in columns by
  their distance from the source, the main chain on one straight line, branches (training
  sequences, channel estimates, meters) in lanes under it, ordered to avoid crossings. A
  long chain wraps into rows that fit the shape of the canvas, alternately left to right
  and right to left (a horseshoe), with the blocks of the reversed rows facing left; scopes
  and meters fed from their bottom sit above the link they tap.
* Links are routed around the blocks; parallel links coming from different outputs that
  share a channel are drawn side by side instead of on top of each other.
* Export .py downloads a standalone script that rebuilds and runs the scenario with the
  Python API, a good starting point for automation.
* The run buttons are disabled while the graph is invalid; the reason is shown next to them.

## Modes

### Live
Simulates one frame per step (batch size 1) at 1–30 steps per second and streams every probe as a scope:
constellation, spectrum, eye diagram, time series, LLR histogram, OFDM grid, channel
response. Parameter changes in the inspector are applied **while running** and the
metrics restart automatically, so moving `awgn.snr_db` or a PA backoff shows the effect
immediately. Structural changes (new blocks or connections) restart the session
automatically. The rate only sets how often the scopes refresh; the metrics accumulate
across frames.

Scope axes carry physical units with SI prefixes (ps, GHz, mA, ...): time and frequency
whenever the sample rate is known, amplitude in √W for optical fields and A for
photocurrents (normalized signals have no unit). Spectra are absolute PSDs (dBm/Hz for
optical fields). The eye diagram is centred on the mean of the traces and spans all of
them; the unit interval is printed in the corner.

### Simulate
Monte-Carlo run in the background: a fixed number of batches or "until a metric reaches
a threshold" (e.g. `ber.bit_errors >= 200`). Shows the final metrics (BER with its 95 %
confidence interval) and the probes of the last batch.

* **frames/batch**: independent frames simulated together per batch (vectorized; saved
  in the scenario as `batch_size`).
* **cores**: worker processes. Batches run in parallel on several CPU cores, each seeded
  from (seed, batch index), so the result does not depend on the number of cores. In a
  sweep, the points run in parallel and give exactly the serial results. The value is
  remembered by the browser.

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
