# Physical-model review (October 2026)

This review checks every physical model in `backend/linkstudio/phy` against standards bodies
(ITU-T, ITU-R, 3GPP/ETSI, IEEE 802.3, ANSI, TIA/ISO) and reference papers and books. Each
entry gives the reference that defines the model, a verdict and, where needed, the action that
would bring the model in line with the reference. Reference keys (`[G.652]`, `[Agrawal]`, ...)
are listed in [references/README.md](../references/README.md); files marked there as
downloaded are in `references/<topic>/`.

Verdicts:

- **OK**: the model and the reference agree, including constants and sign conventions.
- **Simplified**: correct within stated limits; the limit should be documented or lifted.
- **Issue**: disagrees with the reference in a way that changes results.

## Summary

The "Status" column records the follow-up work done after the review. The per-model sections
below keep the original findings, and the [Resolution](#resolution) section describes each fix
and how it is validated.

| Area | Model | Verdict | Status |
|------|-------|---------|--------|
| Baseband | AWGN, Es/N0 and Eb/N0 definitions | OK | Thermal (k T F) mode added |
| RF impairments | Wiener phase noise, CFO, IQ imbalance | OK | - |
| RF impairments | Saleh and Rapp PA models | OK | - |
| RF impairments | Uniform quantizer | OK | - |
| Fading | Rayleigh/Rician flat fading | OK | - |
| Fading | Jakes sum-of-sinusoids generator | Simplified | **Fixed**: Zheng–Xiao generator |
| Fading | EPA/EVA/ETU tapped delay lines | Simplified | **Fixed**: fractional delays, NR TDL-A…E |
| Propagation | Path loss | Missing | **Added**: P.525, 38.901 UMa/UMi/InH, shadowing, LOS probability |
| Copper | Cable insertion loss (√f + f fit) | Simplified | **Fixed**: TIA c/√f term; G.996.1 RLCG loops |
| Copper | Dielectric loss without phase | Issue | **Fixed**: causal Djordjevic–Sarkar dielectric |
| Copper | NEXT/FEXT (ANSI T1.417) | OK | - |
| Copper | 4B/5B, 64B/66B, Manchester, MLT-3 | OK | - |
| Copper | DMT gap and bit loading | OK | - |
| Fibre | GVD, attenuation, split-step NLSE, Manakov | OK | **Fixed**: adaptive step, β3 |
| Fibre | PMD as a fixed DGD | Simplified | **Fixed**: waveplate model, Maxwellian DGD |
| Fibre | Field sign convention | OK | **Documented** (manual §15.2) |
| Amplifier | EDFA ASE and OSNR | OK | - |
| Transmitter | DML/EML intensity modulator | Simplified | **Fixed**: adiabatic chirp, laser bandwidth |
| Transmitter | RIN definition | Issue | **Fixed** |
| Transmitter | IQ Mach–Zehnder modulator | OK | - |
| Receiver | PIN/APD noise | OK | Receiver bandwidth added |
| Receiver | Coherent front end, dual polarization | Issue (minor) | **Fixed**: LO split per polarization |
| Free space | Log-normal and gamma-gamma scintillation | Simplified | **Fixed**: Rytov-based α, β; `fso_channel` with attenuation and pointing errors |
| Short reach | Multimode fibre and VCSEL | New | Implemented |

## Baseband and RF front end

**AWGN** (`channel/awgn.py`): the noise is circular complex Gaussian with N0 per complex
sample, and N0/2 per dimension for real signals. N0 follows the textbook definitions, including
the Ns factor for spatial multiplexing [Proakis §4.1, Tse ch. 3]. **OK.**

**Phase noise**: a Wiener process with increment variance 2πΔν/fs. This is the free-running
oscillator with a Lorentzian line of FWHM Δν [Kikuchi 2016, §IV]. **OK.**

**IQ imbalance**: y = K1 x + K2 x\* with K1 = (1 + g e^{-jφ})/2 and K2 = (1 − g e^{jφ})/2. This
is the standard receiver model [Tse; Tarighat et al. 2005]. **OK.**

**Power amplifiers**:
- Saleh: β_a = 1.1517, α_φ = 4.0033, β_φ = 9.1040 are the TWT fit of [Saleh 1981]. Dividing by
  α_a = 2.1587 only normalizes the small-signal gain to 1.
- Rapp: the AM/AM curve has the published form.

**OK.**

**Quantizer**: a mid-rise uniform ADC with clipping. **OK.**

## Fading and propagation

**Rayleigh/Rician** (`channel/fading.py`):
- Rayleigh: E|h|² = 1.
- Rician: the LOS component √(K/(K+1)) has random phase, and the scattered component has power
  1/(K+1) [Tse §2.4].

**OK.**

**Jakes / Clarke Doppler**: 16 sinusoids with random arrival angles and phases. The ensemble
autocorrelation converges to J0(2πf_D τ) [Clarke 1968, eq. (26)], and the manual verifies this.
**Simplified**: a single realization with random angles has noticeable autocorrelation ripple.
[Zheng & Xiao 2003] give a generator with the correct statistics per realization, at the same
cost.

**Tapped delay lines**:
- The EPA/EVA/ETU delays and powers match [36.104 Annex B.2] exactly.
- **Simplified**: delays are rounded to the sample grid. At LTE sample rates (15.36–30.72 MHz) the
  EPA taps at 30, 70, 90 and 110 ns collapse onto two or three samples, which changes the
  frequency correlation. [38.901 §7.7.2] intends band-limited (sinc-interpolated) fractional taps.
- The 5G NR profiles TDL-A…E and CDL-A…E of [38.901 §7.7] are not implemented yet.
- For historical comparisons, the ITU-R M.1225 models (Pedestrian/Vehicular A/B) are in
  [M.1225].

**Path loss**: nothing models distance yet; only `gain` is available. Recommended additions:
- free-space loss, 20 log10(4πd/λ) [P.525];
- the 38.901 large-scale models UMa, UMi and InH, with shadow fading [38.901 §7.4].

## Copper

**Cable insertion loss** (`wireline.py`, `cable_response`): a fit a√f + b f.
- **Simplified**, for three reasons:
  - The TIA category formulas for Cat 5e and Cat 6A have a third term, c/√f (0.050/√f and
    0.25/√f per 100 m), which the presets omit. It matters below about 1 MHz.
  - Telephone pairs are better described by RLCG primary parameters. Below about 100 kHz the
    loss tends to a constant set by R0, not to zero. This changes the low ADSL tones and any
    echo/NEXT analysis.
  - [G.996.1] and [G.993.2 Annex] define test loops with RLCG (or ABCD) parameter sets. Adding
    the "MAD"/BT/ETSI model with those parameter sets would make DMT rates comparable with the
    standards.
- The skin-effect term carries its exact minimum phase. √(jω) gives equal loss and phase
  slopes, as the manual explains. **OK.**
- **Issue**: the dielectric term b·f is applied as a real attenuation with no phase. Its
  minimum-phase partner (2b/π) f ln f is missing (the causal Djordjevic–Sarkar model
  [Djordjevic et al. 2001]). On FR-4 backplanes, where b·f dominates above a few GHz, this makes
  the impulse response non-causal, with pre-cursor energy. CTLE/DFE results are slightly
  optimistic for the post-cursor and wrong for the pre-cursor. **Action**: derive the phase of
  the loss term by Hilbert transform, or use the Djordjevic–Sarkar permittivity.

**Crosstalk**: the NEXT and FEXT coupling constants are those of [ANSI T1.417] (99 %
worst-case, 1 % model):
- NEXT: 8.536e-15 n^0.6 f^1.5;
- FEXT: 7.744e-21 n^0.6 L_ft f² |H|².

[Starr–Cioffi–Silverman, ch. 3] use the same values. **OK.**

**Line codes**:
- 4B/5B table: IEEE 802.3 clause 24 / FDDI.
- 64B/66B: sync header "01" for data, and the self-synchronous scrambler x^58 + x^39 + 1 of
  IEEE 802.3 clause 49.2.6.
- Manchester polarity: IEEE 802.3 clause 7.
- MLT-3: ANSI X3.263.

**OK.**

**DMT**: the tone spacing 4.3125 kHz, the 512-point IFFT and the 9.8 dB gap for uncoded QAM at
1e-7 follow [G.992.1] and [Cioffi ch. 4]. **OK.** Possible extensions: trellis coding gain and
the G.992.1 bit-swap algorithm.

## Optical fibre and amplifiers

**Dispersion and attenuation** (`optical.py`):
- β2 = −Dλ²/(2πc) and the all-pass operator exp(jβ2Lω²/2) are [Agrawal NLFO, eq. 3.2.5].
- Default values (17 ps/(nm km), 0.2 dB/km) match [G.652] fibre at 1550 nm.
- **OK.** Suggested addition: the dispersion slope β3. It matters for WDM grids wider than about
  1 THz [G.694.1], and for transmission near the zero-dispersion wavelength.

**Split-step Fourier**:
- The symmetric scheme puts the nonlinear step at the midpoint, with the exact effective length
  2 sinh(α dz/2)/α. The Manakov factor is 8/9 [Marcuse et al. 1997]. **OK.**
- The O(h²) convergence is verified in the manual.
- **Simplified**: the default step of 2 km leaves a 12 % error on a 28 dBm soliton.
  [Sinkin 2003] recommends the nonlinear-phase criterion (step chosen so that γ P dz stays
  below about 5 mrad), or the local-error method. **Action**: make the nonlinear-phase step the
  default.

**PMD**: a random unitary rotation plus a DGD of exactly PMD·√L per section. **Simplified**: in
real fibre the first-order DGD is Maxwellian with that mean [G.652, §6.6; Kogelnik et al.].
Concatenating spans in `optical_link` already gives some statistics, but a single span always
has the mean DGD. **Action**: draw the DGD from a Maxwellian distribution, or use a
coarse-step waveplate model of 10 or more sections per span.

**Sign convention**:
- The code uses Agrawal's convention: carrier e^{-iω0t}, dispersion exp(+jβ2ω²z/2) and
  nonlinear phase exp(+jγ|A|²z). It does so consistently, so solitons, SPM, CD compensation and
  the LO offset all behave correctly.
- With NumPy's FFT, a positive baseband frequency is therefore a *lower* optical frequency.
- **Action**: state this in the manual's notation section. It matters only when channel indices
  are mapped onto the [G.694.1] grid, or when walk-off direction is compared with a textbook.

**EDFA**: ASE PSD per polarization n_sp hν(G−1) with n_sp = NF·G/(2(G−1)), and OSNR in
12.5 GHz (0.1 nm). The definitions are those of [G.662]/[G.661] and [Agrawal FOCS ch. 7]; the
manual reproduces the 58 dB rule within 0.05 dB. **OK.** Possible extensions: gain saturation and
spectral gain tilt (multi-channel WDM), and the GN model for nonlinear interference
[Poggiolini 2014].

## Optical transmitters and receivers

**Intensity modulator**:
- The ER-to-modulation-depth mapping is correct.
- The transient chirp, phase = (α/2) ln P, is the Henry-factor relation [Koch & Bowers 1984].
- **Simplified**:
  - The adiabatic chirp term κP is missing; it dominates the frequency excursion of DMLs.
  - The laser modulation bandwidth is ignored: the output follows the drive instantly.
- **Action**: add the adiabatic chirp and a second-order relaxation-oscillation response. The
  new `vcsel` block already implements the response.

**RIN**:
- IEEE 802.3 and [Personick 1973] define RIN relative to the *average* power:
  σ_P² = RIN · P_avg² · B.
- The old implementation scaled the noise by the instantaneous power.
- **Fixed in this review**: `intensity_noise()` in `optical.py` is shared by
  `intensity_modulator` and `vcsel`, and now uses the average-power definition.

**IQ MZM**: push-pull at null bias, E ∝ sin(πV/(2Vπ)). **OK.**

**Photodiode**:
- Shot noise 2qM²F I B with McIntyre's excess-noise factor F = kM + (1−k)(2 − 1/M)
  [McIntyre 1966].
- Thermal noise as an input-referred current density [Personick 1973].
- **OK.**
- The noise bandwidth was always fs/2, so the result depended on oversampling unless a receive
  filter followed. **Added in this review**: the optional `bandwidth_ghz` applies the 4th-order
  Bessel–Thomson reference receiver used in IEEE 802.3 optical PMD specifications. The reported
  noise variance is scaled by the filter's noise bandwidth.

**Coherent front end**: signal R√P_LO·E and LO shot noise 2qRP_LO·B per complex output, as for
the balanced 90° hybrid of [Kikuchi 2016, §III]. **Issue (minor)**:
- With dual polarization, a real receiver splits the LO between two hybrids, so each
  polarization sees P_LO/2.
- The code gives each polarization the full P_LO. That overstates the signal by 3 dB relative to
  thermal noise; the shot-noise-limited SNR is unaffected.
- **Action**: use P_LO/n_pol per polarization.

**FSO**:
- Log-normal with E[h] = 1 (ln h ~ N(−σ²/2, σ²)), and gamma-gamma as the product of two
  unit-mean gamma variates, as in [Al-Habash et al. 2001]. **OK.**
- **Simplified**:
  - α and β are free parameters, whereas [Al-Habash et al. 2001] give them as functions of the
    Rytov variance, i.e. of Cn², wavelength and distance.
  - Atmospheric attenuation is absent: Kim/Kruse visibility model [P.1814], fog and rain
    [P.1817].
  - Pointing errors are absent [Farid & Hranilovic 2007].
- **Action**: add a `fso_channel` block that computes all three from link geometry and weather
  (see [Khalighi & Uysal 2014], [Kaushal & Kaddoum 2015]).

## Short-reach optics (new in this review)

`phy/multimode.py` adds `multimode_fiber` (OM1–OM5) and `vcsel`.

**Fibre classes**: core 50 µm (62.5 µm for OM1), NA 0.200 (0.275). Overfilled and effective
modal bandwidths at 850 nm follow ISO/IEC 11801 / TIA-568.3: OM3 1500/2000 and OM4 3500/4700
MHz·km. Cabled attenuation is 3.5 dB/km [G.651.1; IEEE 802.3 clauses 52 and 95].

**`bandwidth` model**: the IEEE 802.3 link model [Cunningham & Dawe 2002] (file in
`references/optical/multimode/`):
- Gaussian modal response with bandwidth EMB/L;
- chromatic RMS width σ_λ√(D1² + D2²), with D1 = S0λ/4·(1 − (λ0/λ)⁴) and D2 = 0.7 S0σ_λ;
- the Gaussian constant B·σ = 0.187.

**`mode_groups` model**: WKB theory of α-profile fibres [Gloge & Marcatili 1973; Olshansky &
Keck 1976]:
- mode count M = α/(α+2)·V²/2, about √M principal groups;
- delay of group μ from the second-order expansion with profile dispersion ε;
- overfilled launch with power ∝ (2μ − 1);
- Gaussian launches (centre or offset) through Hermite–Gauss overlap integrals;
- without an explicit α, the profile is detuned from α_opt so that the overfilled bandwidth
  equals the class specification.

`tests/test_multimode.py` checks the model against closed forms:
- α_opt and σ_min = L N1 Δ²/(20√3 c) of Olshansky–Keck;
- the WKB delay;
- continuum convergence of the discrete groups;
- Poisson group powers for an offset launch and geometric even-group powers for a centred,
  mismatched launch (both exact for a parabolic core);
- the IEEE chromatic term;
- variance additivity of pulse spreading.

**VCSEL**: small-signal response f_r²/(f_r² − f² + jfγ/2π) times a parasitic pole
[Coldren ch. 5; Larsson 2011]. It adds RIN and publishes the spectral width for the fibre.

**Scenario**: `scenarios/optical/datacenter_vcsel_mmf.yaml`, a 25.78 GBd NRZ lane of a
100GBASE-SR4-like link.

**Known limits**, as follow-ups:
- mode-partition noise and modal noise (penalties in the IEEE link model);
- coupling between modal and chromatic dispersion (wavelength-dependent group delays
  [Golowich et al. 2003]);
- DMD templates from TIA-455-220 / IEC 60793-1-49 for statistically realistic fibres;
- a nonlinear VCSEL rate-equation model for large-signal PAM-4.

## Resolution

Every item marked Issue or Simplified has been fixed. All fixes are covered by tests (217 in
total) and described in the manual as theory, implementation and validation.

| Item | Fix | Validation |
|------|-----|------------|
| Dielectric loss | `dielectric_response`: Djordjevic–Sarkar permittivity (ω1 = 1e4, ω2 = 1e12 rad/s), loss tangent matched to `b`, front delay removed | Loss within 1 % of b·f from 10 MHz to 1 GHz; energy before t = 0 on the backplane falls from 40 % to 5e-7 |
| Cable loss fit | TIA c/√f term for Cat 5e and Cat 6A (constant below 1 MHz) | Insertion loss equals the TIA formula |
| Telephone loops | `pic_26awg` and `pic_24awg`: G.996.1 Annex A primary constants (70 °F), ABCD two-port between 100 Ω; the ADSL scenario now uses them | Mid-C loop within 0.35 dB of G.996.1 Table 2 from 20 kHz to 1.1 MHz (manual Fig. 14.3) |
| Coherent LO | Each polarization's hybrid receives P_LO / n_pol | Beat amplitude R·√(P_LO/2)·\|E\| |
| PMD | `pmd_transfer`: waveplate model with N sections (`pmd_sections`, default 10) of DGD ⟨τ⟩√(3π/8N) | 4000 fibres: mean DGD within 1 %; ⟨τ²⟩/⟨τ⟩² = 1.168 against 3π/8 = 1.178 (manual Fig. 15.6) |
| Dispersion slope | β3 from S (`dispersion_slope_ps_nm2_km`); accumulated in `cd_slope_ps_nm2`, removed by `cd_compensation` | Group delay D·L·Δλ + S·L/2·Δλ² within 0.05 ps; CD+slope compensation restores the pulse |
| Split-step step size | Nonlinear-phase criterion (`nonlinear_phase_rad` = 0.05 rad), longest step 1/`steps_per_km` | Soliton error 3e-4 with the defaults (12 % with fixed 2 km steps) |
| Sign convention | Documented in the manual (NLSE section): positive NumPy frequency = lower optical frequency | Covered by the β3 group-delay test |
| DML | `adiabatic_chirp_ghz_mw` (Koch–Bowers adiabatic term); `resonance_ghz` / `damping_per_ns` / `parasitic_ghz` small-signal bandwidth with group delay in `delay` | A power step shifts the frequency by κ·ΔP |
| Fading generator | Zheng–Xiao sum of sinusoids | ACF = J0 within 0.02 with 8 sinusoids; uncorrelated I/Q |
| Tapped delay lines | Band-limited (Hann-windowed sinc) fractional taps (`fractional_delays`); TR 38.901 TDL-A…E scaled by `delay_spread_ns`; Ricean LOS tap for D/E | Normalized RMS delay spread = 1 for every profile; TDL-D K1 = 13.3 dB ± 1 dB; kernel delay within 5 % up to 0.75 π |
| Path loss | New `path_loss` block: free space (P.525), UMa / UMi / InH (TR 38.901 Table 7.4.1-1), log-normal shadowing, LOS probability (Table 7.4.2-1) | P.525 value; continuity at the breakpoint; NLOS ≥ LOS; shadowing σ and LOS fraction (manual Fig. 9.2) |
| Absolute noise | `awgn` `thermal` mode, N0 = k T F | Noise variance k T F f_s; free-space QPSK link budget BER matches theory |
| FSO | `fso_turbulence.rytov_variance`; new `fso_channel`: Kim attenuation (P.1814-1), geometric loss, scintillation from Cn², pointing errors (Farid–Hranilovic) | α, β at σ_R² = 1 (4.39, 2.56); scintillation index; Kim and geometric formulas; mean pointing loss |

New scenarios: `wireless/link_budget_umi.yaml` and `optical/fso_link_budget.yaml`.

**Remaining known limits** (also listed in the manual):
- large-signal laser rate equations;
- mode-partition and modal noise in multimode links;
- time-correlated FSO fading, aperture averaging and beam wander;
- spatially consistent (CDL / geometry-based) channel models;
- EDFA gain dynamics and the GN model.
