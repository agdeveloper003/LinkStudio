"""Large-scale propagation: free-space and 3GPP TR 38.901 path loss with shadow fading."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from linkstudio.core import Block, BlockError, BlockParams, Kind, Port, RunContext, Signal, register_block

C0 = 3.0e8  # TR 38.901 uses c = 3.0e8 m/s

# Default antenna heights (m) and validity range of the 2-D distance (m) per scenario.
SCENARIO_DEFAULTS: dict[str, tuple[float, float, tuple[float, float]]] = {
    "uma": (25.0, 1.5, (10.0, 5000.0)),
    "umi": (10.0, 1.5, (10.0, 5000.0)),
    "inh": (3.0, 1.0, (1.0, 150.0)),
}


def free_space_loss_db(d3d_m: np.ndarray, fc_ghz: float) -> np.ndarray:
    """Free-space basic transmission loss 20 log10(4 pi d f / c) (ITU-R P.525)."""
    return 20 * np.log10(4 * np.pi * np.asarray(d3d_m) * fc_ghz * 1e9 / 299_792_458.0)


def tr38901_path_loss_db(scenario: str, los: np.ndarray, d2d_m: np.ndarray, fc_ghz: float,
                         h_bs: float, h_ut: float) -> tuple[np.ndarray, np.ndarray]:
    """Path loss (dB) and shadow-fading standard deviation (dB) of TR 38.901 Table 7.4.1-1.

    UMa and UMi-Street Canyon use the dual-slope LOS model with the effective breakpoint
    d'_BP = 4 h'_BS h'_UT f_c / c (h_E = 1 m, i.e. UMa with h_UT < 13 m); NLOS is the maximum
    of the LOS and NLOS formulas. InH is the office model.
    """
    d2d = np.asarray(d2d_m, dtype=float)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    lf, ld = np.log10(fc_ghz), np.log10(d3d)
    if scenario in ("uma", "umi"):
        dbp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / C0
        if scenario == "uma":
            pl1 = 28.0 + 22 * ld + 20 * lf
            pl2 = 28.0 + 40 * ld + 20 * lf - 9 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
            nlos = 13.54 + 39.08 * ld + 20 * lf - 0.6 * (h_ut - 1.5)
            sigma = (4.0, 6.0)
        else:
            pl1 = 32.4 + 21 * ld + 20 * lf
            pl2 = 32.4 + 40 * ld + 20 * lf - 9.5 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
            nlos = 35.3 * ld + 22.4 + 21.3 * lf - 0.3 * (h_ut - 1.5)
            sigma = (4.0, 7.82)
        pl_los = np.where(d2d <= dbp, pl1, pl2)
    else:
        pl_los = 32.4 + 17.3 * ld + 20 * lf
        nlos = 38.3 * ld + 17.30 + 24.9 * lf
        sigma = (3.0, 8.03)
    pl = np.where(los, pl_los, np.maximum(pl_los, nlos))
    return pl, np.where(los, sigma[0], sigma[1])


def los_probability(scenario: str, d2d_m: np.ndarray, h_ut: float) -> np.ndarray:
    """LOS probability of TR 38.901 Table 7.4.2-1 (UMa, UMi-Street Canyon, InH-Mixed office)."""
    d = np.asarray(d2d_m, dtype=float)
    if scenario == "inh":
        return np.where(d <= 1.2, 1.0, np.where(d < 6.5, np.exp(-(d - 1.2) / 4.7),
                                                 0.32 * np.exp(-(d - 6.5) / 32.6)))
    if scenario == "umi":
        return np.where(d <= 18, 1.0, 18 / d + np.exp(-d / 36) * (1 - 18 / d))
    c = 0.0 if h_ut <= 13 else ((h_ut - 13) / 10) ** 1.5
    base = 18 / d + np.exp(-d / 63) * (1 - 18 / d)
    return np.where(d <= 18, 1.0, base * (1 + c * 5 / 4 * (d / 100) ** 3 * np.exp(-d / 150)))


@register_block("path_loss", category="channel")
class PathLoss(Block):
    """Distance-dependent path loss and log-normal shadow fading (one value per frame).

    * ``free_space``: ITU-R P.525.
    * ``uma`` / ``umi`` / ``inh``: 3GPP TR 38.901 urban macro, urban micro street canyon and
      indoor office, LOS, NLOS or LOS drawn with the probability of Table 7.4.2-1.

    The amplitude is scaled by 10^(-(PL + SF)/20) and the ``power`` / ``noise_var`` metadata
    follow, like ``gain``. Use it with an absolute power level and ``awgn`` in ``thermal`` mode
    for link budgets (mapper power 1 = 1 W = 30 dBm; set the transmit power with ``gain``).
    """

    class Params(BlockParams):
        model: Literal["free_space", "uma", "umi", "inh"] = "free_space"
        frequency_ghz: float = Field(3.5, gt=0, description="Carrier frequency.")
        distance_m: float = Field(100.0, gt=0, description="Ground (2-D) distance between the antennas.")
        h_bs_m: float | None = Field(None, gt=0, description="Base-station height (default per scenario).")
        h_ut_m: float | None = Field(None, gt=0, description="User height (default per scenario).")
        condition: Literal["los", "nlos", "probabilistic"] = "probabilistic"
        shadow_fading: bool = Field(True, description="Log-normal shadowing per frame.")

    inputs = {"in": Port(Kind.SYMBOLS | Kind.SAMPLES)}
    outputs = {"out": Port.like("in")}

    def on_reset(self) -> None:
        self._loss: list[float] = []
        self._los: list[float] = []

    def draw_loss_db(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Total loss (dB) and LOS flags for ``n`` independent frames."""
        p = self.params
        if p.model == "free_space":
            h_bs, h_ut = p.h_bs_m or 0.0, p.h_ut_m or 0.0
            d3d = np.hypot(p.distance_m, h_bs - h_ut)
            return np.full(n, float(free_space_loss_db(d3d, p.frequency_ghz))), np.ones(n, dtype=bool)
        h_bs0, h_ut0, (dmin, dmax) = SCENARIO_DEFAULTS[p.model]
        h_bs, h_ut = p.h_bs_m or h_bs0, p.h_ut_m or h_ut0
        if p.condition == "los":
            los = np.ones(n, dtype=bool)
        elif p.condition == "nlos":
            los = np.zeros(n, dtype=bool)
        else:
            los = self.rng.random(n) < los_probability(p.model, p.distance_m, h_ut)
        pl, sigma = tr38901_path_loss_db(p.model, los, np.full(n, p.distance_m), p.frequency_ghz, h_bs, h_ut)
        if p.shadow_fading:
            pl = pl + sigma * self.rng.standard_normal(n)
        return pl, los

    def setup(self) -> None:
        p = self.params
        if p.model in SCENARIO_DEFAULTS:
            dmin, dmax = SCENARIO_DEFAULTS[p.model][2]
            if not dmin <= p.distance_m <= dmax:
                raise BlockError(f"{p.model} is specified for {dmin:g} m <= d <= {dmax:g} m.")

    def process(self, inputs: dict[str, Signal], ctx: RunContext) -> dict[str, Signal]:
        x = inputs["in"]
        B = x.data.shape[0]
        loss_db, los = self.draw_loss_db(B)
        self._loss.extend(loss_db.tolist())
        self._los.extend(los.astype(float).tolist())
        g = 10 ** (-loss_db / 20)
        shape = (B,) + (1,) * (x.data.ndim - 1)
        meta = {}
        for key in ("noise_var", "power"):
            if x.meta.get(key) is not None:
                meta[key] = x.meta[key] * float(np.mean(g ** 2))
        return {"out": x.evolve(x.data * g.reshape(shape), **meta)}

    def metrics(self) -> dict[str, float]:
        if not self._loss:
            return {}
        return {"path_loss_db": float(np.mean(self._loss)), "los_fraction": float(np.mean(self._los))}
