# LinkStudio Engine — Architecture (v0.1, PHY layer)

This document describes the Python simulation engine as implemented in `backend/linkstudio`.
For the overall product vision (UI, modes, roadmap) see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Design goals

| Goal | How the engine meets it |
|------|-------------------------|
| Modular | Everything is a **block** with typed ports; chains are **graphs** of blocks; graphs can be wrapped as **subsystems** and nested. |
| Flexible | Any topology (branches, fan-out, several receivers on one transmitter), any parameter changeable at run time, YAML scenarios. |
| Correct | Pure DSP functions separated from blocks, 81 unit/physics tests, demo scripts that compare against closed-form theory. |
| Reproducible | One master seed; every block derives its own random stream from `(seed, block path)`. |
| Fast enough | Batch-first: every block processes `(batch, n)` arrays with vectorized NumPy. |
| Layer-agnostic core | `linkstudio.core` knows nothing about the PHY. MAC/network blocks can later reuse graph, simulation, sweeps and scenarios as they are. |

---

## 2. Package layout

```
backend/
├── linkstudio/
│   ├── core/                  # layer-agnostic engine
│   │   ├── signal.py          # Signal (data + kind + fs + metadata), Kind flags
│   │   ├── block.py           # Block base class, Port, BlockParams (Pydantic)
│   │   ├── registry.py        # @register_block, create_block, block_catalogue
│   │   ├── graph.py           # Graph: connect, validate, topological execution
│   │   ├── composite.py       # Subsystem: a graph exposed as a block
│   │   ├── simulation.py      # Simulation (Monte-Carlo driver), StopCriteria
│   │   ├── sweep.py           # Sweep (parameter grids), SweepResult (table/CSV)
│   │   ├── scenario.py        # YAML <-> Graph, Scenario runner
│   │   └── context.py         # RunContext passed to blocks
│   ├── phy/                   # physical-layer library
│   │   ├── dsp/               # pure functions (no blocks)
│   │   │   ├── bits.py            # bit/int conversion, Gray, LLR helpers
│   │   │   ├── constellations.py  # PSK/QAM/PAM/custom, hard & soft demapping
│   │   │   ├── filters.py         # RRC/RC/Gaussian/rect pulses, FIR, fractional delay
│   │   │   ├── lfsr.py            # PRBS / scrambler sequences
│   │   │   └── analysis.py        # PSD, PAPR, EVM, eye traces
│   │   ├── sources.py         # random_bits, prbs
│   │   ├── coding/            # crc, scrambler, interleaver, repetition, hamming, conv + viterbi
│   │   ├── modulation.py      # mapper, demapper
│   │   ├── waveform/          # pulse_shaper, matched_filter, up/downsampler, fir_filter, OFDM
│   │   ├── channel/           # awgn, flat_fading, multipath, impairments
│   │   ├── receiver.py        # one_tap / linear (ZF, MMSE) / LMS equalizers
│   │   ├── sync.py            # preamble, timing / frame / carrier recovery
│   │   ├── mimo.py            # spatial multiplexing, Alamouti, MIMO detector
│   │   ├── wireline.py        # line codes, cables, crosstalk, SerDes EQ, DMT
│   │   ├── optical.py         # lasers, fiber, EDFA, photodiode, coherent DSP
│   │   ├── metrics.py         # ber_meter, evm_meter, power_meter, optical_power_meter
│   │   └── theory.py          # closed-form BER references
│   ├── viz.py                 # matplotlib helpers (constellation, PSD, eye, BER)
│   └── cli.py                 # python -m linkstudio blocks | describe | run
├── scripts/                   # runnable demos (main.py + one per topic)
├── scenarios/                 # YAML scenarios and reusable subsystems
└── tests/                     # pytest suite
```

Dependency direction is strict: `phy` depends on `core`, never the reverse;
`phy/dsp` depends on nothing but NumPy/SciPy.

---

## 3. Core concepts

### 3.1 Signal

Every edge carries a `Signal`:

```python
Signal(data: np.ndarray, kind: Kind, fs: float | None, meta: dict)
```

* **Array convention** — axis 0 is the batch (independent frames simulated in
  parallel), the last axis is time. OFDM grids are `(batch, n_ofdm_symbols, n_fft)`.
* **Kind** — semantic type, a flag enum: `BITS`, `LLR`, `SYMBOLS`, `SAMPLES`, `GRID`,
  `CSI`, `FLAGS`. Ports declare which kinds they accept/produce, so wrong
  connections are rejected before running.
* **Metadata** — travels downstream and lets blocks configure themselves
  consistently without duplicating parameters:

| key | set by | used by |
|-----|--------|---------|
| `bits_per_symbol`, `constellation` | mapper | AWGN (Eb/N0), demapper, LMS |
| `code_rate` | encoders, CRC | AWGN (Eb/N0) |
| `sps`, `pulse_taps`, `delay` | pulse shaper, OFDM | AWGN (Es/N0), matched filter |
| `power` | mapper, shaper, OFDM, gain, multipath | AWGN nominal SNR reference |
| `noise_var` | AWGN, matched filter, equalizers | soft demapper |
| `ofdm` | OFDM modulator | demodulator, estimator, equalizer |
| `csi_type` | fading blocks | equalizers, OFDM estimator |

  Example: the demapper needs no parameters — it reads the constellation from the
  mapper and the noise variance from the AWGN block (as modified by the matched filter
  and the equalizer).

### 3.2 Block

```python
@register_block("awgn", category="channel")
class AWGN(Block):
    class Params(BlockParams):            # Pydantic: validated, serializable, JSON Schema
        snr_db: float = 10.0
        mode: Literal["snr", "esn0", "ebn0"] = "ebn0"

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}    # output kind mirrors the input

    def process(self, inputs, ctx):       # one batch in, one batch out
        ...
```

Lifecycle:

| method | when | purpose |
|--------|------|---------|
| `setup()` | construction and every `set_param` | derived state (filter taps, trellis, tables) |
| `reset(seed, path)` | start of a simulation | new reproducible RNG, then `on_reset()` |
| `on_reset()` | inside `reset` | clear accumulators/state |
| `process(inputs, ctx)` | every batch | the computation |
| `metrics()` | anytime | accumulated measurements (metric blocks) |

Because parameters are Pydantic models, `block_catalogue()` already exports
every block's ports and JSON Schema — the future UI can generate its forms from it.

### 3.3 Graph

```python
g = Graph("link")
g.create("src", "random_bits", n_bits=2000)
g.create("map", "mapper", scheme="qam", order=16)
...
g.chain("src", "map", "awgn", "demap", "ber.rx")   # default ports
g.connect("src", "ber.tx")                          # explicit ports
```

* Addresses: `"block.port"`, `"block.param"`, and `"/"` to enter subsystems
  (`"rx/demap.method"`).
* `validate()` checks unconnected required inputs, cycles and kind compatibility
  (resolving `Port.like` outputs), then caches a topological order.
* `execute(ctx)` runs one batch and returns every produced signal (`"block.port"`),
  including subsystem internals (`"tx/map.out"`) — this is what probes/scopes use.
* Feedback loops live **inside** blocks (e.g. the LMS equalizer), keeping the graph
  a DAG that is simple to schedule.

### 3.4 Subsystem (hierarchy)

```python
tx = Subsystem(tx_graph, outputs={"bits": "src.out", "symbols": "map.out"})
rx = Subsystem(rx_graph, inputs={"in": "demap.in"}, outputs={"bits": "dec.out"})
top.add("tx", tx); top.add("rx", rx)
```

A subsystem is a block whose ports map onto inner ports. Its random streams,
parameters, metrics and captured signals are namespaced by path (`rx/ber.ber`).
This is the mechanism that will scale to network simulations: a "node" will be a
subsystem containing PHY (and later MAC) graphs.

### 3.5 Drivers

| driver | purpose |
|--------|---------|
| `Simulation.step()` | one batch, returns all signals → future *live mode* |
| `Simulation.run(n_batches | stop, capture)` | Monte-Carlo run with `StopCriteria` (min errors, max batches, max time) |
| `Sweep(graph, {"awgn.snr_db": [...]}, mode="grid" | "zip")` | parameter sweeps, common random numbers, restores parameters afterwards |
| `Scenario.run()` | runs what a YAML file describes (simulation or sweep) |

### 3.6 Scenarios (YAML)

```yaml
blocks:
  tx:   {type: subsystem, file: subsystems/coded_qam_tx.yaml}
  awgn: {type: awgn, params: {mode: ebn0, snr_db: 6.0}}
  rx:   {type: subsystem, file: subsystems/coded_qam_rx.yaml}
  ber:  {type: ber_meter}
edges:
  - tx.symbols -> awgn.in
  - awgn.out -> rx.in
  - rx.bits -> ber.rx
  - tx.bits -> ber.tx
simulation: {n_batches: 10}
```

`save_scenario` / `load_scenario` round-trip any graph (tested to reproduce identical
error counts).

---

## 4. PHY block library (86 blocks)

| category | blocks |
|----------|--------|
| source | `random_bits`, `prbs` |
| coding | `crc_encoder/decoder` (CRC-8/16/24A/32), `scrambler`, `interleaver` (block/random, inverse), `repetition_encoder/decoder`, `hamming_encoder/decoder`, `conv_encoder`, `viterbi_decoder` (soft/hard), `puncturer` / `depuncturer`, `ldpc_encoder/decoder` (QC-LDPC, min-sum / sum-product BP), `polar_encoder/decoder` (SC decoding) |
| modulation | `mapper` (PSK/QAM/PAM/custom, Gray), `demapper` (hard, max-log, exact LLR) |
| waveform | `pulse_shaper` (RRC/RC/Gaussian/rect), `matched_filter` (optionally non-decimating), `upsampler`, `downsampler`, `fir_filter`, `ofdm_modulator`, `ofdm_demodulator` |
| channel | `awgn` (SNR/Es/N0/Eb/N0), `flat_fading` (Rayleigh/Rician; block/fast/Jakes), `multipath` (TDL, 3GPP EPA/EVA/ETU; block or time-varying Jakes taps), `mimo_channel` (Nr x Nt Rayleigh, Kronecker correlation), `gain`, `carrier_offset`, `phase_noise`, `timing_offset`, `iq_imbalance`, `power_amplifier` (Rapp/Saleh/clip), `quantizer` |
| sync | `ofdm_sync` (blind CP-based timing + fractional CFO, van de Beek), `preamble` (Schmidl & Cox structure + guard), `timing_recovery` (Oerder & Meyr init + Gardner loop + cubic interpolator), `frame_sync` (preamble detection, CFO estimate/correction, phase), `carrier_recovery` (decision-directed PLL) |
| mimo | `stream_demux` / `stream_mux` (spatial multiplexing), `alamouti_encoder` / `alamouti_decoder` |
| receiver | `one_tap_equalizer`, `linear_equalizer` (ZF/MMSE FIR from CIR), `lms_equalizer` (NLMS, training + decision directed), `ofdm_channel_estimator` (LS + interpolation / perfect, also time-varying), `ofdm_equalizer`, `mimo_detector` (ZF / MMSE / ML) |
| metrics | `ber_meter` (BER with 95 % Wilson confidence interval, SER, FER), `evm_meter` (EVM/MER), `power_meter` (power/PAPR), `optical_power_meter` (dBm, OSNR) |
| wireline | `line_encoder` / `line_decoder` (NRZ, Manchester, AMI, MLT-3), `line_block_encoder` / `line_block_decoder` (4B/5B, 64B/66B), `copper_line` (Cat5e/6A, 0.4/0.5 mm pairs, RG-6, PCB backplane), `crosstalk` (ANSI NEXT/FEXT), `impulse_noise`, `hybrid_echo`, `echo_canceller` (NLMS), `ctle`, `dfe_equalizer` (adaptive FFE + DFE), `jitter` (RJ + SJ), `dmt_modulator` / `dmt_demodulator` (water-filling bit loading) |
| optical | `intensity_modulator` (ER, RIN, chirp), `iq_modulator` (linear / MZM, dual polarization), `fiber` (loss, dispersion, split-step Kerr, PMD), `optical_amplifier` (EDFA, ASE), `optical_link` (multi-span), `wdm_multiplexer` / `wdm_demultiplexer` (WDM with real neighbour channels: XPM/FWM in nonlinear fiber), `fso_turbulence` (log-normal, gamma-gamma), `photodiode` (PIN/APD, shot + thermal noise), `coherent_receiver` (LO, 90-degree hybrid), `cd_compensation` |
| coherent DSP | `fo_compensation` (4th-power, polarization-robust), `adaptive_mimo_equalizer` (2x2 butterfly, CMA / trained LMS), `phase_recovery` (BPS, Viterbi-Viterbi), `agc` |

`python -m linkstudio blocks` lists them; `python -m linkstudio describe <block>` prints
ports and parameters.

### Conventions worth knowing

* Constellations have unit average energy; LLR = log P(b=0)/P(b=1).
* Pulses have unit energy; the matched filter rescales so that symbols come out at
  nominal amplitude and updates `noise_var`.
* AWGN: N0 = P·sps / (Es/N0), and Eb/N0 additionally divides by k·R. P is the
  **nominal** power from metadata by default (average-SNR definition); use
  `power_reference: measured` or `signal_power` to override.
* OFDM uses a unitary FFT, so Es/N0 refers to the per-subcarrier SNR (CP and pilot
  overhead excluded). Pilots are a fixed pseudo-random QPSK sequence.
* Fading blocks output true CSI on a `csi` port: equalizers/estimators can use it
  (genie benchmark) or ignore it (pilot-based estimation).
* MIMO signals have shape `(batch, streams, n)`. The MIMO channel is normalized by
  1/sqrt(Nt): total transmit power is constant and the AWGN SNR is per receive
  antenna; with spatial multiplexing Eb/N0 accounts for the `spatial_rate` metadata.
* Synchronization chain: `preamble -> pulse_shaper -> channel -> matched_filter(decimate=false)
  -> timing_recovery -> frame_sync -> carrier_recovery`. Implementation loss is about
  0.3 dB at BER 1e-3 with a fractional delay, a few kHz CFO and phase noise.
* Wireline signals are real-valued; the AWGN block then adds real noise of variance N0/2,
  so NRZ / PAM BER follow the usual theory. Cable models use loss
  `a*sqrt(f_MHz) + b*f_MHz` dB/km with the skin-effect phase term (long, realistic tails).
  DMT uses the ADSL downstream tone plan (tones 33-255) and the gap approximation.
* Optical signals are field envelopes in sqrt(W). Physical noise: ASE (`n_sp h nu (G-1)` per
  polarization, tracked as `ase_psd` for OSNR), shot noise `2 q I B`, thermal noise from an
  input-referred density; `agc` brings photocurrents back to symbol scale.
* Coherent chain: `iq_modulator -> optical_link -> coherent_receiver -> cd_compensation ->
  fo_compensation -> matched_filter(decimate=false) -> agc -> adaptive_mimo_equalizer ->
  phase_recovery`. Penalty vs OSNR theory is ~0.3 dB. Frame-based processing uses circular
  FFTs, so the receiver makes the LO phase periodic over each frame, and the FO block removes
  the bulk delay that an offset causes after dispersion compensation.

---

## 5. Validation

`pytest` (172 tests, ~15 s) covers:

* **core** — validation errors, reproducibility, stop criteria, captures, sweeps,
  subsystems, YAML round trip, catalogue;
* **dsp** — Gray labelling, noiseless demapping, Nyquist property of RRC⊗RRC, PRBS
  period/balance, LFSR continuation;
* **coding** — CRC detection, scrambler/interleaver inverses, Hamming single-error
  correction, Viterbi decoding and soft-decision gain;
* **channels** — noise variance for every SNR mode, fading statistics, Jakes
  correlation, TDL presets, neutrality of impairments at zero settings;
* **physics** — BER vs closed-form theory for BPSK/QPSK/8-PSK/16-QAM/64-QAM/4-PAM,
  RRC-shaped links, Rayleigh+ZF, OFDM over AWGN, OFDM over EPA, ISI equalizers,
  nominal power tracking;
* **advanced** — synchronized receiver vs theory (delay + CFO + phase noise), LDPC
  parity checks and decoding, polar SC decoding, puncturing, 1x2 MRC / Alamouti /
  2x2 ZF vs diversity theory, detector ordering ML < MMSE < ZF, time-varying TDL;
* **api** — catalogue, scenario load/save (no path traversal), validation, simulation
  and sweep jobs, live WebSocket with parameter changes;
* **wireline / optical** — line codes and block codes, 64B/66B error multiplication, NRZ
  with real noise vs theory, cable loss vs TIA-568, SerDes CTLE + DFE, water-filling DMT,
  crosstalk, echo cancellation, jitter; fiber loss, Gaussian-pulse dispersion broadening
  and its exact compensation, SPM phase, EDFA OSNR (58 + Pin - NF), OOK IM/DD vs Q-factor
  theory, PAM-4 dispersion penalty, coherent DP-QPSK vs OSNR theory, FSO scintillation,
  MLT-3, WDM cross-phase modulation, OFDM timing/CFO sync;
* **scenarios** — every scenario file of the library loads, validates and runs.

`scripts/main.py` runs all demos and reports 34 theory-based checks.

---

## 6. Known limitations (next candidates)

* Polar decoding is plain SC (no SCL / CRC-aided list decoding); LDPC uses a random
  QC construction (no standard 5G/Wi-Fi base graphs yet).
* MIMO is flat-fading single-carrier (no MIMO-OFDM yet); no precoding/beamforming.
* OFDM synchronization (CP-based timing/CFO) is not implemented; single-carrier sync is.
* Decision-directed loops show the usual threshold effect below ~3 dB Eb/N0.
* LMS, Viterbi, polar SC and the sync loops iterate in Python (fine for 10^3–10^4 symbols).
* Wireline: no TEQ for DMT (the ADSL tone plan avoids the long cable tail), no vectoring,
  no Tomlinson-Harashima precoding; 8B/10B is not implemented (64B/66B and 4B/5B are).
* Optical: WDM interferers are generated around one channel under test (no per-channel
  receivers), first-order PMD only, no probabilistic shaping.
* OFDM sync estimates fractional CFO only (no integer CFO); the LS estimator removes the
  common delay slope before interpolating, valid up to n_fft / (2 x pilot spacing) samples.

---

## 7. User interface

`python -m linkstudio serve` starts a FastAPI server (`linkstudio/api`) that serves the React
UI (`frontend/`, built into `frontend/dist`) on http://127.0.0.1:8000.

| API | purpose |
|-----|---------|
| `GET /api/blocks` | block catalogue with ports and JSON Schemas (drives palette and forms) |
| `GET/PUT /api/scenarios/{path}` | load / save YAML scenarios (layout and probes included) |
| `POST /api/validate` | build + validate a graph, resolved port kinds, subsystem ports |
| `POST /api/layout` | automatic block layout |
| `POST /api/export/python` | standalone Python script equivalent to the graph |
| `POST /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` | background simulations and sweeps with progress |
| `WS /api/live` | live mode: batch loop, probe streaming, `set_param` while running |

Probes (`linkstudio/api/probes.py`) turn any signal into compact plot data: constellation,
spectrum, eye diagram, time series, LLR histogram, OFDM grid heatmap, CSI views.
See [GUI.md](GUI.md) for the user guide.
