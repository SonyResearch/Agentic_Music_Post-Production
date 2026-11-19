import torch
from torchcomp import coef2ms, ms2coef
from torch import nn

from ...param_decoding import DecodedParamsMixin
from .itomaster_functional import (
    distortion,
    fast_limiter,
    limiter,
    multiband_compressor,
    fast_multiband_compressor,
)


class Distortion_itomaster(nn.Module, DecodedParamsMixin):
    def __init__(
        self,
        sample_rate: int = 48000,
        min_gain_db: float = 0.0,
        max_gain_db: float = 24.0,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)

    @staticmethod
    def _node_param(p: torch.Tensor, n_nodes: int):
        return p.view(n_nodes, -1)[:, 0]

    def _minmax(self, x: torch.Tensor):
        return self.min_gain_db + (self.max_gain_db - self.min_gain_db) * torch.sigmoid(x)

    def forward(self, x, drive_db):
        n_nodes = x.shape[0]
        drive_db = self._node_param(drive_db, n_nodes)
        drive_db = self._minmax(drive_db)
        return distortion(x, drive_db=drive_db)

    def parameter_size(self):
        return {"drive_db": 1}

    def decode_params(self, raw_params):
        return {"drive_db": self._minmax(raw_params["drive_db"])}

    def decode_param_metadata(self, raw_params=None):
        return {
            "drive_db": {
                "display_name": "drive_db",
                "unit": "db",
                "kind": "gain",
            }
        }


class Multiband_Compressor_itomaster(nn.Module, DecodedParamsMixin):
    def __init__(
        self,
        sample_rate: int = 48000,
        min_low_cutoff_hz: float = 20.0,
        max_low_cutoff_hz: float = 300.0,
        min_high_cutoff_hz: float = 2000.0,
        max_high_cutoff_hz: float = 12000.0,
        min_threshold_db_comp: float = -60.0,
        max_threshold_db_comp: float = -1e-6,
        min_ratio_comp: float = 1.0 + 1e-6,
        max_ratio_comp: float = 20.0,
        low_band_min_attack_ms_comp: float = 20.0,
        low_band_max_attack_ms_comp: float = 120.0,
        low_band_min_release_ms_comp: float = 120.0,
        low_band_max_release_ms_comp: float = 600.0,
        mid_band_min_attack_ms_comp: float = 5.0,
        mid_band_max_attack_ms_comp: float = 50.0,
        mid_band_min_release_ms_comp: float = 60.0,
        mid_band_max_release_ms_comp: float = 300.0,
        high_band_min_attack_ms_comp: float = 0.5,
        high_band_max_attack_ms_comp: float = 15.0,
        high_band_min_release_ms_comp: float = 20.0,
        high_band_max_release_ms_comp: float = 150.0,
        min_threshold_db_exp: float = -60.0,
        max_threshold_db_exp: float = -1e-6,
        min_ratio_exp: float = 1e-6,
        max_ratio_exp: float = 1.0 - 1e-6,
        low_band_min_attack_ms_exp: float = 10.0,
        low_band_max_attack_ms_exp: float = 80.0,
        low_band_min_release_ms_exp: float = 150.0,
        low_band_max_release_ms_exp: float = 700.0,
        mid_band_min_attack_ms_exp: float = 5.0,
        mid_band_max_attack_ms_exp: float = 40.0,
        mid_band_min_release_ms_exp: float = 80.0,
        mid_band_max_release_ms_exp: float = 350.0,
        high_band_min_attack_ms_exp: float = 1.0,
        high_band_max_attack_ms_exp: float = 15.0,
        high_band_min_release_ms_exp: float = 40.0,
        high_band_max_release_ms_exp: float = 180.0,
        expander_enabled: bool = False,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
        fast_multiband_compressor: bool = False,
        target_control_rate_hz: float = 3000.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.min_low_cutoff_hz = float(min_low_cutoff_hz)
        self.max_low_cutoff_hz = float(max_low_cutoff_hz)
        self.min_high_cutoff_hz = float(min_high_cutoff_hz)
        self.max_high_cutoff_hz = float(max_high_cutoff_hz)
        self.min_threshold_db_comp = float(min_threshold_db_comp)
        self.max_threshold_db_comp = float(max_threshold_db_comp)
        self.min_ratio_comp = float(min_ratio_comp)
        self.max_ratio_comp = float(max_ratio_comp)
        self.min_threshold_db_exp = float(min_threshold_db_exp)
        self.max_threshold_db_exp = float(max_threshold_db_exp)
        self.min_ratio_exp = float(min_ratio_exp)
        self.max_ratio_exp = float(max_ratio_exp)
        self.comp_ranges_ms = {
            "low_shelf": {"at": (float(low_band_min_attack_ms_comp), float(low_band_max_attack_ms_comp)), "rt": (float(low_band_min_release_ms_comp), float(low_band_max_release_ms_comp))},
            "mid_band": {"at": (float(mid_band_min_attack_ms_comp), float(mid_band_max_attack_ms_comp)), "rt": (float(mid_band_min_release_ms_comp), float(mid_band_max_release_ms_comp))},
            "high_shelf": {"at": (float(high_band_min_attack_ms_comp), float(high_band_max_attack_ms_comp)), "rt": (float(high_band_min_release_ms_comp), float(high_band_max_release_ms_comp))},
        }
        self.exp_ranges_ms = {
            "low_shelf": {"at": (float(low_band_min_attack_ms_exp), float(low_band_max_attack_ms_exp)), "rt": (float(low_band_min_release_ms_exp), float(low_band_max_release_ms_exp))},
            "mid_band": {"at": (float(mid_band_min_attack_ms_exp), float(mid_band_max_attack_ms_exp)), "rt": (float(mid_band_min_release_ms_exp), float(mid_band_max_release_ms_exp))},
            "high_shelf": {"at": (float(high_band_min_attack_ms_exp), float(high_band_max_attack_ms_exp)), "rt": (float(high_band_min_release_ms_exp), float(high_band_max_release_ms_exp))},
        }
        self.expander_enabled = bool(expander_enabled)
        self.fast_multiband_compressor = bool(fast_multiband_compressor)
        self.target_control_rate_hz = float(target_control_rate_hz)

    @staticmethod
    def _node_param(p: torch.Tensor, n_nodes: int):
        return p.view(n_nodes, -1)[:, 0]

    @staticmethod
    def _minmax(x: torch.Tensor, lo: float, hi: float):
        return lo + (hi - lo) * torch.sigmoid(x)

    def _bounded_smoothing_coef(self, x: torch.Tensor, min_ms: float, max_ms: float):
        min_ms_t = x.new_full((), float(min_ms))
        max_ms_t = x.new_full((), float(max_ms))
        coef_a = ms2coef(min_ms_t, self.sample_rate)
        coef_b = ms2coef(max_ms_t, self.sample_rate)
        coef_lo = torch.minimum(coef_a, coef_b)
        coef_hi = torch.maximum(coef_a, coef_b)
        return self._minmax(x, coef_lo, coef_hi)

    def _decode_params(self, x, **params):
        n_nodes = x.shape[0]
        decoded = {}
        for key, value in params.items():
            if value is None:
                continue
            decoded[key] = self._node_param(value, n_nodes)

        decoded["low_cutoff"] = self._minmax(
            decoded["low_cutoff"], self.min_low_cutoff_hz, self.max_low_cutoff_hz
        )
        decoded["high_cutoff"] = self._minmax(
            decoded["high_cutoff"], self.min_high_cutoff_hz, self.max_high_cutoff_hz
        )

        for band in ("low_shelf", "mid_band", "high_shelf"):
            decoded[f"{band}_comp_thresh"] = self._minmax(
                decoded[f"{band}_comp_thresh"],
                self.min_threshold_db_comp,
                self.max_threshold_db_comp,
            )
            decoded[f"{band}_comp_ratio"] = self._minmax(
                decoded[f"{band}_comp_ratio"],
                self.min_ratio_comp,
                self.max_ratio_comp,
            )
            comp_at_lo, comp_at_hi = self.comp_ranges_ms[band]["at"]
            comp_rt_lo, comp_rt_hi = self.comp_ranges_ms[band]["rt"]
            if self.fast_multiband_compressor:
                decoded[f"{band}_comp_at"] = self._bounded_smoothing_coef(
                    decoded[f"{band}_comp_at"], comp_at_lo, comp_at_hi
                )
                decoded[f"{band}_comp_rt"] = self._bounded_smoothing_coef(
                    decoded[f"{band}_comp_rt"], comp_rt_lo, comp_rt_hi
                )
            else:
                decoded[f"{band}_comp_at"] = self._minmax(
                    decoded[f"{band}_comp_at"], comp_at_lo, comp_at_hi
                )
                decoded[f"{band}_comp_rt"] = self._minmax(
                    decoded[f"{band}_comp_rt"], comp_rt_lo, comp_rt_hi
                )
            if self.expander_enabled:
                decoded[f"{band}_exp_thresh"] = self._minmax(
                    decoded[f"{band}_exp_thresh"],
                    self.min_threshold_db_exp,
                    self.max_threshold_db_exp,
                )
                decoded[f"{band}_exp_ratio"] = self._minmax(
                    decoded[f"{band}_exp_ratio"],
                    self.min_ratio_exp,
                    self.max_ratio_exp,
                )
                exp_at_lo, exp_at_hi = self.exp_ranges_ms[band]["at"]
                exp_rt_lo, exp_rt_hi = self.exp_ranges_ms[band]["rt"]
                if self.fast_multiband_compressor:
                    decoded[f"{band}_exp_at"] = self._bounded_smoothing_coef(
                        decoded[f"{band}_exp_at"], exp_at_lo, exp_at_hi
                    )
                    decoded[f"{band}_exp_rt"] = self._bounded_smoothing_coef(
                        decoded[f"{band}_exp_rt"], exp_rt_lo, exp_rt_hi
                    )
                else:
                    decoded[f"{band}_exp_at"] = self._minmax(
                        decoded[f"{band}_exp_at"], exp_at_lo, exp_at_hi
                    )
                    decoded[f"{band}_exp_rt"] = self._minmax(
                        decoded[f"{band}_exp_rt"], exp_rt_lo, exp_rt_hi
                    )
            else:
                decoded[f"{band}_exp_thresh"] = decoded[f"{band}_comp_thresh"].new_full(
                    decoded[f"{band}_comp_thresh"].shape, -120.0
                )
                decoded[f"{band}_exp_ratio"] = decoded[f"{band}_comp_ratio"].new_ones(
                    decoded[f"{band}_comp_ratio"].shape
                )
                if self.fast_multiband_compressor:
                    unit_coef = decoded[f"{band}_comp_ratio"].new_ones(
                        decoded[f"{band}_comp_ratio"].shape
                    )
                    decoded[f"{band}_exp_at"] = unit_coef
                    decoded[f"{band}_exp_rt"] = unit_coef
                else:
                    decoded[f"{band}_exp_at"] = decoded[f"{band}_comp_at"]
                    decoded[f"{band}_exp_rt"] = decoded[f"{band}_comp_rt"]

        return decoded

    def forward(
        self,
        x,
        low_cutoff,
        high_cutoff,
        low_shelf_comp_thresh,
        low_shelf_comp_ratio,
        low_shelf_comp_at,
        low_shelf_comp_rt,
        mid_band_comp_thresh,
        mid_band_comp_ratio,
        mid_band_comp_at,
        mid_band_comp_rt,
        high_shelf_comp_thresh,
        high_shelf_comp_ratio,
        high_shelf_comp_at,
        high_shelf_comp_rt,
        *,
        low_shelf_exp_thresh=None,
        low_shelf_exp_ratio=None,
        low_shelf_exp_at=None,
        low_shelf_exp_rt=None,
        mid_band_exp_thresh=None,
        mid_band_exp_ratio=None,
        mid_band_exp_at=None,
        mid_band_exp_rt=None,
        high_shelf_exp_thresh=None,
        high_shelf_exp_ratio=None,
        high_shelf_exp_at=None,
        high_shelf_exp_rt=None,
    ):
        decoded = self._decode_params(
            x,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            low_shelf_comp_thresh=low_shelf_comp_thresh,
            low_shelf_comp_ratio=low_shelf_comp_ratio,
            low_shelf_exp_thresh=low_shelf_exp_thresh,
            low_shelf_exp_ratio=low_shelf_exp_ratio,
            low_shelf_comp_at=low_shelf_comp_at,
            low_shelf_comp_rt=low_shelf_comp_rt,
            low_shelf_exp_at=low_shelf_exp_at,
            low_shelf_exp_rt=low_shelf_exp_rt,
            mid_band_comp_thresh=mid_band_comp_thresh,
            mid_band_comp_ratio=mid_band_comp_ratio,
            mid_band_exp_thresh=mid_band_exp_thresh,
            mid_band_exp_ratio=mid_band_exp_ratio,
            mid_band_comp_at=mid_band_comp_at,
            mid_band_comp_rt=mid_band_comp_rt,
            mid_band_exp_at=mid_band_exp_at,
            mid_band_exp_rt=mid_band_exp_rt,
            high_shelf_comp_thresh=high_shelf_comp_thresh,
            high_shelf_comp_ratio=high_shelf_comp_ratio,
            high_shelf_exp_thresh=high_shelf_exp_thresh,
            high_shelf_exp_ratio=high_shelf_exp_ratio,
            high_shelf_comp_at=high_shelf_comp_at,
            high_shelf_comp_rt=high_shelf_comp_rt,
            high_shelf_exp_at=high_shelf_exp_at,
            high_shelf_exp_rt=high_shelf_exp_rt,
        )
        compressor_fn = (
            fast_multiband_compressor if self.fast_multiband_compressor else multiband_compressor
        )
        kwargs = {"sample_rate": self.sample_rate, "expander_enabled": self.expander_enabled, **decoded}
        if self.fast_multiband_compressor:
            kwargs["target_control_rate_hz"] = self.target_control_rate_hz
        return compressor_fn(x, **kwargs)

    def parameter_size(self):
        size = {
            "low_cutoff": 1,
            "high_cutoff": 1,
            "low_shelf_comp_thresh": 1,
            "low_shelf_comp_ratio": 1,
            "low_shelf_comp_at": 1,
            "low_shelf_comp_rt": 1,
            "mid_band_comp_thresh": 1,
            "mid_band_comp_ratio": 1,
            "mid_band_comp_at": 1,
            "mid_band_comp_rt": 1,
            "high_shelf_comp_thresh": 1,
            "high_shelf_comp_ratio": 1,
            "high_shelf_comp_at": 1,
            "high_shelf_comp_rt": 1,
        }
        if self.expander_enabled:
            size.update(
                {
                    "low_shelf_exp_thresh": 1,
                    "low_shelf_exp_ratio": 1,
                    "low_shelf_exp_at": 1,
                    "low_shelf_exp_rt": 1,
                    "mid_band_exp_thresh": 1,
                    "mid_band_exp_ratio": 1,
                    "mid_band_exp_at": 1,
                    "mid_band_exp_rt": 1,
                    "high_shelf_exp_thresh": 1,
                    "high_shelf_exp_ratio": 1,
                    "high_shelf_exp_at": 1,
                    "high_shelf_exp_rt": 1,
                }
            )
        return size

    def decode_params(self, raw_params):
        probe = next(iter(raw_params.values()))
        n_nodes = int(probe.shape[0]) if probe.ndim > 0 else 1
        x = probe.reshape(n_nodes, -1)[:, :1]
        return self._decode_params(x, **raw_params)


class Limiter_itomaster(nn.Module, DecodedParamsMixin):
    def __init__(
        self,
        sample_rate: int = 48000,
        min_threshold_db: float = -60.0,
        max_threshold_db: float = -1e-6,
        min_attack_ms: float = 0.5,
        max_attack_ms: float = 10.0,
        min_release_ms: float = 20.0,
        max_release_ms: float = 300.0,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
        fast_limiter: bool = False,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.min_threshold_db = float(min_threshold_db)
        self.max_threshold_db = float(max_threshold_db)
        self.min_attack_ms = float(min_attack_ms)
        self.max_attack_ms = float(max_attack_ms)
        self.min_release_ms = float(min_release_ms)
        self.max_release_ms = float(max_release_ms)
        self.fast_limiter = bool(fast_limiter)

    @staticmethod
    def _node_param(p: torch.Tensor, n_nodes: int):
        return p.view(n_nodes, -1)[:, 0]

    @staticmethod
    def _minmax(x: torch.Tensor, lo: float, hi: float):
        return lo + (hi - lo) * torch.sigmoid(x)

    def _bounded_smoothing_coef(self, x: torch.Tensor, min_ms: float, max_ms: float):
        min_ms_t = x.new_full((), float(min_ms))
        max_ms_t = x.new_full((), float(max_ms))
        coef_a = ms2coef(min_ms_t, self.sample_rate)
        coef_b = ms2coef(max_ms_t, self.sample_rate)
        coef_lo = torch.minimum(coef_a, coef_b)
        coef_hi = torch.maximum(coef_a, coef_b)
        return self._minmax(x, coef_lo, coef_hi)

    def decode_attack_release_coef(self, at_raw: torch.Tensor, rt_raw: torch.Tensor):
        return (
            self._bounded_smoothing_coef(at_raw, self.min_attack_ms, self.max_attack_ms),
            self._bounded_smoothing_coef(rt_raw, self.min_release_ms, self.max_release_ms),
        )

    def attack_release_ms(self, at_coef: torch.Tensor, rt_coef: torch.Tensor):
        return (
            coef2ms(at_coef, self.sample_rate),
            coef2ms(rt_coef, self.sample_rate),
        )

    def forward(self, x, threshold, at, rt):
        n_nodes = x.shape[0]
        threshold = self._minmax(
            self._node_param(threshold, n_nodes),
            self.min_threshold_db,
            self.max_threshold_db,
        )
        at, rt = self.decode_attack_release_coef(
            self._node_param(at, n_nodes),
            self._node_param(rt, n_nodes),
        )
        limiter_fn = fast_limiter if self.fast_limiter else limiter
        return limiter_fn(
            x,
            sample_rate=self.sample_rate,
            threshold=threshold,
            at=at,
            rt=rt,
        )

    def parameter_size(self):
        return {"threshold": 1, "at": 1, "rt": 1}

    def decode_params(self, raw_params):
        threshold = self._minmax(
            raw_params["threshold"],
            self.min_threshold_db,
            self.max_threshold_db,
        )
        at = self._bounded_smoothing_coef(
            raw_params["at"],
            self.min_attack_ms,
            self.max_attack_ms,
        )
        rt = self._bounded_smoothing_coef(
            raw_params["rt"],
            self.min_release_ms,
            self.max_release_ms,
        )
        at_ms, rt_ms = self.attack_release_ms(at, rt)
        return {
            "threshold": threshold,
            "at": at,
            "rt": rt,
            "at_ms": at_ms,
            "rt_ms": rt_ms,
        }
