import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torchcomp import ms2coef, coef2ms, db2amp
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from typing import List, Tuple, Union, Any, Optional, Callable
import math
from torch_fftconv import fft_conv1d
from functools import reduce

from ...param_decoding import DecodedParamsMixin
from .diffvox_functional import (
    compressor_expander,
    lowpass_biquad,
    highpass_biquad,
    equalizer_biquad,
    lowshelf_biquad,
    highshelf_biquad,
    lowpass_biquad_coef,
    highpass_biquad_coef,
    highshelf_biquad_coef,
    lowshelf_biquad_coef,
    equalizer_biquad_coef,
)
from .diffvox_utils import chain_functions


class Clip(nn.Module):
    def __init__(self, max: Optional[float] = None, min: Optional[float] = None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        if self.min is not None:
            x = torch.clip(x, min=self.min)
        if self.max is not None:
            x = torch.clip(x, max=self.max)
        return x


def clip_delay_eq_Q(m: nn.Module, Q: float):
    if isinstance(m, Delay) and isinstance(m.eq, LowPass):
        register_parametrization(m.eq.params, "Q", Clip(max=Q))
    return m


float2param = lambda x: nn.Parameter(
    torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
)

STEREO_NORM = math.sqrt(2)


def broadcast2stereo(m, args):
    x, *_ = args
    return x.expand(-1, 2, -1) if x.shape[1] == 1 else x


hadamard = lambda x: torch.stack([x.sum(1), x[:, 0] - x[:, 1]], 1) / STEREO_NORM


class Hadamard(nn.Module):
    def forward(self, x):
        return hadamard(x)


class FX(nn.Module, DecodedParamsMixin):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.params = nn.ParameterDict({k: float2param(v) for k, v in kwargs.items()})


class SmoothingCoef(nn.Module):
    def forward(self, x):
        return x.sigmoid()

    def right_inverse(self, y):
        return (y / (1 - y)).log()


class CompRatio(nn.Module):
    def forward(self, x):
        return x.exp() + 1

    def right_inverse(self, y):
        return torch.log(y - 1)


class MinMax(nn.Module):
    def __init__(self, min=0.0, max: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(min, torch.Tensor):
            self.register_buffer("min", min, persistent=False)
        else:
            self.min = min

        if isinstance(max, torch.Tensor):
            self.register_buffer("max", max, persistent=False)
        else:
            self.max = max

        self._m = SmoothingCoef()

    def forward(self, x):
        return self._m(x) * (self.max - self.min) + self.min

    def right_inverse(self, y):
        return self._m.right_inverse((y - self.min) / (self.max - self.min))


class WrappedPositive(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

    def forward(self, x):
        return x.abs() % self.period

    def right_inverse(self, y):
        return y


class CompressorExpander(FX):
    cmp_ratio_min: float = 1
    cmp_ratio_max: float = 20

    def __init__(
        self,
        sr: int,
        cmp_ratio: float = 2.0,
        exp_ratio: float = 0.5,
        at_ms: float = 50.0,
        rt_ms: float = 50.0,
        avg_coef: float = 0.3,
        cmp_th: float = -18.0,
        exp_th: float = -54.0,
        make_up: float = 0.0,
        delay: int = 0,
        lookahead: bool = False,
        max_lookahead: float = 15.0,
    ):
        super().__init__(
            cmp_th=cmp_th,
            exp_th=exp_th,
            make_up=make_up,
            avg_coef=avg_coef,
            cmp_ratio=cmp_ratio,
            exp_ratio=exp_ratio,
        )
        # deprecated, please use lookahead instead
        self.delay = delay
        self.sr = sr

        self.params["at"] = nn.Parameter(ms2coef(torch.tensor(at_ms), sr))
        self.params["rt"] = nn.Parameter(ms2coef(torch.tensor(rt_ms), sr))

        if lookahead:
            self.params["lookahead"] = nn.Parameter(torch.ones(1) / sr * 1000)
            register_parametrization(
                self.params, "lookahead", WrappedPositive(max_lookahead)
            )
            sinc_length = int(sr * (max_lookahead + 1) * 0.001) + 1
            left_pad_size = int(sr * 0.001)
            self._pad_size = (left_pad_size, sinc_length - left_pad_size - 1)
            self.register_buffer(
                "_arange",
                torch.arange(sinc_length) - left_pad_size,
                persistent=False,
            )
        self.lookahead = lookahead

        register_parametrization(self.params, "at", SmoothingCoef())
        register_parametrization(self.params, "rt", SmoothingCoef())
        register_parametrization(self.params, "avg_coef", SmoothingCoef())
        register_parametrization(
            self.params, "cmp_ratio", MinMax(self.cmp_ratio_min, self.cmp_ratio_max)
        )
        register_parametrization(self.params, "exp_ratio", SmoothingCoef())

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"attack: {coef2ms(self.params.at, self.sr).item()} (ms)\n"
                f"release: {coef2ms(self.params.rt, self.sr).item()} (ms)\n"
                f"avg_coef: {self.params.avg_coef.item()}\n"
                f"compressor_ratio: {self.params.cmp_ratio.item()}\n"
                f"expander_ratio: {self.params.exp_ratio.item()}\n"
                f"compressor_threshold: {self.params.cmp_th.item()} (dB)\n"
                f"expander_threshold: {self.params.exp_th.item()} (dB)\n"
                f"make_up: {self.params.make_up.item()} (dB)"
            )
            if self.lookahead:
                s += f"\nlookahead: {self.params.lookahead.item()} (ms)"
        return s

    def forward(self, x):
        if self.lookahead:
            lookahead_in_samples = self.params.lookahead * 0.001 * self.sr
            sinc_filter = torch.sinc(self._arange - lookahead_in_samples)
            lookahead_func = lambda gain: F.conv1d(
                F.pad(
                    gain.view(-1, 1, gain.size(-1)), self._pad_size, mode="replicate"
                ),
                sinc_filter[None, None, :],
            ).view(*gain.shape)
        else:
            lookahead_func = lambda x: x

        return compressor_expander(
            x.reshape(-1, x.shape[-1]),
            lookahead_func=lookahead_func,
            **{k: v for k, v in self.params.items() if k != "lookahead"},
        ).view(*x.shape)


class Panning(FX):
    def __init__(self, pan: float = 0.0):
        assert pan <= 100 and pan >= -100
        super().__init__(pan=(pan + 100) / 200)

        register_parametrization(self.params, "pan", SmoothingCoef())

        self.register_forward_pre_hook(broadcast2stereo)

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"pan: {self.params.pan.item() * 200 - 100}"
        return s

    def forward(self, x: torch.Tensor):
        angle = self.params.pan.view(1) * torch.pi * 0.5
        amp = torch.concat([angle.cos(), angle.sin()]).view(2, 1) * STEREO_NORM
        return x * amp


class StereoWidth(Panning):
    def forward(self, x: torch.Tensor):
        return chain_functions(hadamard, super().forward, hadamard)(x)


class ImpulseResponse(nn.Module):
    def forward(self, h):
        return torch.cat([torch.ones_like(h[..., :1]), h], dim=-1)


class FIR(FX):
    def __init__(
        self,
        length: int,
        channels: int = 2,
        conv_method: str = "direct",
    ):
        super().__init__(kernel=torch.zeros(channels, length - 1))
        self._padding = length - 1
        self.channels = channels

        match conv_method:
            case "direct":
                self.conv_func = F.conv1d
            case "fft":
                self.conv_func = fft_conv1d
            case _:
                raise ValueError(f"Unknown conv_method: {conv_method}")

        if channels == 2:
            self.register_forward_pre_hook(broadcast2stereo)

    def forward(self, x: torch.Tensor):
        zero_padded = F.pad(x[..., :-1], (self._padding, 0), "constant", 0)
        return x + self.conv_func(
            zero_padded, self.params.kernel.flip(1).unsqueeze(1), groups=self.channels
        )


class QFactor(nn.Module):
    def forward(self, x):
        return x.exp()

    def right_inverse(self, y):
        return y.log()


class LowPass(FX):
    def __init__(
        self,
        sr: int,
        freq: float = 17500.0,
        Q: float = 0.707,
        min_freq: float = 200.0,
        max_freq: float = 18000,
        min_Q: float = 0.5,
        max_Q: float = 10.0,
    ):
        super().__init__(freq=freq, Q=Q)

        self.sr = sr
        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))
        register_parametrization(self.params, "Q", MinMax(min_Q, max_Q))

    def forward(self, x):
        return lowpass_biquad(
            x, sample_rate=self.sr, cutoff_freq=self.params.freq, Q=self.params.Q
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, Q: {self.params.Q.item():.4f}"
        return s


class HighPass(LowPass):
    def __init__(
        self,
        *args,
        freq: float = 200.0,
        min_freq: float = 16.0,
        max_freq: float = 5300.0,
        **kwargs,
    ):
        super().__init__(
            *args, freq=freq, min_freq=min_freq, max_freq=max_freq, **kwargs
        )

    def forward(self, x):
        return highpass_biquad(
            x, sample_rate=self.sr, cutoff_freq=self.params.freq, Q=self.params.Q
        )


class Peak(FX):
    def __init__(
        self,
        sr: int,
        gain: float = 0.0,
        freq: float = 2000.0,
        Q: float = 0.707,
        min_freq: float = 33.0,
        max_freq: float = 17500.0,
        min_Q: float = 0.2,
        max_Q: float = 20,
    ):
        super().__init__(freq=freq, Q=Q, gain=gain)

        self.sr = sr

        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))
        register_parametrization(self.params, "Q", MinMax(min_Q, max_Q))

    def forward(self, x):
        return equalizer_biquad(
            x,
            sample_rate=self.sr,
            center_freq=self.params.freq,
            Q=self.params.Q,
            gain=self.params.gain,
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, gain: {self.params.gain.item():.4f}, Q: {self.params.Q.item():.4f}"
        return s


class LowShelf(FX):
    def __init__(
        self,
        sr: int,
        gain: float = 0.0,
        freq: float = 115.0,
        min_freq: float = 30,
        max_freq: float = 200,
    ):
        super().__init__(freq=freq, gain=gain)

        self.sr = sr
        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))

        self.register_buffer("Q", torch.tensor(0.707), persistent=False)

    def forward(self, x):
        return lowshelf_biquad(
            x,
            sample_rate=self.sr,
            cutoff_freq=self.params.freq,
            gain=self.params.gain,
            Q=self.Q,
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, gain: {self.params.gain.item():.4f}"
        return s


class HighShelf(LowShelf):
    def __init__(
        self,
        *args,
        freq: float = 4525,
        min_freq: float = 750,
        max_freq: float = 8300,
        **kwargs,
    ):
        super().__init__(
            *args, freq=freq, min_freq=min_freq, max_freq=max_freq, **kwargs
        )

    def forward(self, x):
        return highshelf_biquad(
            x,
            sample_rate=self.sr,
            cutoff_freq=self.params.freq,
            gain=self.params.gain,
            Q=self.Q,
        )


def module2coeffs(
    m: Union[LowPass, HighPass, Peak, LowShelf, HighShelf],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    match m:
        case LowPass():
            return lowpass_biquad_coef(m.sr, m.params.freq, m.params.Q)
        case HighPass():
            return highpass_biquad_coef(m.sr, m.params.freq, m.params.Q)
        case Peak():
            return equalizer_biquad_coef(m.sr, m.params.freq, m.params.Q, m.params.gain)
        case LowShelf():
            return lowshelf_biquad_coef(m.sr, m.params.freq, m.params.gain, m.Q)
        case HighShelf():
            return highshelf_biquad_coef(m.sr, m.params.freq, m.params.gain, m.Q)
        case _:
            raise ValueError(f"Unknown module: {m}")


class AlwaysNegative(nn.Module):
    def forward(self, x):
        return -F.softplus(x)

    def right_inverse(self, y):
        return torch.log(y.neg().exp() - 1)


class Reverb(FX):
    def __init__(self, ir_len=60000, n_fft=384, hop_length=192, downsample_factor=1,
                 flashfftconv=False, max_input_len=2**17):
        super().__init__(
            log_mag=torch.full((2, n_fft // downsample_factor // 2 + 1), -1.0),
            log_mag_delta=torch.full((2, n_fft // downsample_factor // 2 + 1), -5.0),
        )

        self.steps = (ir_len - n_fft + hop_length - 1) // hop_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.downsample_factor = downsample_factor

        self._noise_angle = nn.Parameter(
            torch.rand(2, n_fft // 2 + 1, self.steps) * 2 * torch.pi
        )

        self.register_buffer(
            "_arange", torch.arange(self.steps, dtype=torch.float32), persistent=False
        )
        self.spec_forward = Spectrogram(n_fft, hop_length=hop_length, power=None)
        self.spec_inverse = InverseSpectrogram(
            n_fft,
            hop_length=hop_length,
        )

        register_parametrization(self.params, "log_mag", AlwaysNegative())
        register_parametrization(self.params, "log_mag_delta", AlwaysNegative())

        self.register_forward_pre_hook(broadcast2stereo)

    def forward(self, x, log_mag=None, log_mag_delta=None):
        h = x
        H = self.spec_forward(h)

        log_mag = self.params.log_mag
        log_mag_delta = self.params.log_mag_delta

        if self.downsample_factor > 1:
            log_mag = F.interpolate(
                log_mag.unsqueeze(0),
                size=self._noise_angle.size(1),
                align_corners=True,
                mode="linear",
            ).squeeze(0)
            log_mag_delta = F.interpolate(
                log_mag_delta.unsqueeze(0),
                size=self._noise_angle.size(1),
                align_corners=True,
                mode="linear",
            ).squeeze(0)

        ir_2d = torch.exp(
            log_mag.unsqueeze(-1)
            + log_mag_delta.unsqueeze(-1) * self._arange
            + self._noise_angle * 1j
        )

        padded_H = F.pad(H.flatten(1, 2), (ir_2d.shape[-1] - 1, 0))

        H = F.conv1d(
            padded_H,
            hadamard(ir_2d.unsqueeze(0)).flatten(1, 2).flip(-1).transpose(0, 1),
            groups=H.shape[2] * 2,
        ).view(*H.shape)

        h = self.spec_inverse(H)
        return h


class ParametricEQ_diffvox(FX):
    def __init__(
        self,
        sample_rate: int = 30000,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
    ):
        super().__init__(
            peak1_freq=800.0,
            peak1_gain=0.0,
            peak1_Q=0.707,
            peak2_freq=4000.0,
            peak2_gain=0.0,
            peak2_Q=0.707,
            lowshelf_freq=115.0,
            lowshelf_gain=0.0,
            highshelf_freq=6000.0,
            highshelf_gain=0.0,
            lowpass_freq=17500.0,
            lowpass_Q=0.707,
            highpass_freq=200.0,
            highpass_Q=0.707,
        )
        self.sr = sample_rate
        self.register_buffer("_shelf_q_scalar", torch.tensor(0.707), persistent=False)

    def _smoothing_coef(self, x):
        return x.sigmoid()

    def _minmax(self, x, lo: float, hi: float):
        return lo + (hi - lo) * x.sigmoid()

    def _node_param_to_nc(self, p, N, C):
        # Graph params arrive as [N, 1] (or compatible); keep the first slot per node.
        p_n = p.view(N, -1)[:, 0]
        # Expand per node to channels without materializing repeats first.
        return p_n[:, None].expand(N, C).reshape(N * C)

    def forward(
        self,
        x,
        peak1_freq,
        peak1_gain,
        peak1_Q,
        peak2_freq,
        peak2_gain,
        peak2_Q,
        lowshelf_freq,
        lowshelf_gain,
        highshelf_freq,
        highshelf_gain,
        lowpass_freq,
        lowpass_Q,
        highpass_freq,
        highpass_Q,
    ):
        N, C, T = x.shape
        x_flat = x.reshape(N * C, T)

        peak1_freq = self._minmax(peak1_freq, 33.0, 5400.0)
        peak2_freq = self._minmax(peak2_freq, 200.0, 17500.0)
        lowshelf_freq = self._minmax(lowshelf_freq, 30.0, 200.0)
        highshelf_freq = self._minmax(highshelf_freq, 750.0, 8300.0)
        lowpass_freq = self._minmax(lowpass_freq, 200.0, 18000.0)
        highpass_freq = self._minmax(highpass_freq, 16.0, 5300.0)

        peak1_Q = self._minmax(peak1_Q, 0.2, 20.0)
        peak2_Q = self._minmax(peak2_Q, 0.2, 20.0)
        lowpass_Q = self._minmax(lowpass_Q, 0.5, 10.0)
        highpass_Q = self._minmax(highpass_Q, 0.5, 10.0)

        peak1_freq = self._node_param_to_nc(peak1_freq, N, C)
        peak1_gain = self._node_param_to_nc(peak1_gain, N, C)
        peak1_Q = self._node_param_to_nc(peak1_Q, N, C)
        peak2_freq = self._node_param_to_nc(peak2_freq, N, C)
        peak2_gain = self._node_param_to_nc(peak2_gain, N, C)
        peak2_Q = self._node_param_to_nc(peak2_Q, N, C)
        lowshelf_freq = self._node_param_to_nc(lowshelf_freq, N, C)
        lowshelf_gain = self._node_param_to_nc(lowshelf_gain, N, C)
        highshelf_freq = self._node_param_to_nc(highshelf_freq, N, C)
        highshelf_gain = self._node_param_to_nc(highshelf_gain, N, C)
        lowpass_freq = self._node_param_to_nc(lowpass_freq, N, C)
        lowpass_Q = self._node_param_to_nc(lowpass_Q, N, C)
        highpass_freq = self._node_param_to_nc(highpass_freq, N, C)
        highpass_Q = self._node_param_to_nc(highpass_Q, N, C)

        shelf_Q = self._shelf_q_scalar.to(lowshelf_freq.device, lowshelf_freq.dtype).expand_as(lowshelf_freq)

        y = equalizer_biquad(
            x_flat, sample_rate=self.sr, center_freq=peak1_freq, Q=peak1_Q, gain=peak1_gain
        )
        y = equalizer_biquad(
            y, sample_rate=self.sr, center_freq=peak2_freq, Q=peak2_Q, gain=peak2_gain
        )
        y = lowshelf_biquad(
            y, sample_rate=self.sr, cutoff_freq=lowshelf_freq, gain=lowshelf_gain, Q=shelf_Q
        )
        y = highshelf_biquad(
            y, sample_rate=self.sr, cutoff_freq=highshelf_freq, gain=highshelf_gain, Q=shelf_Q
        )
        y = lowpass_biquad(y, sample_rate=self.sr, cutoff_freq=lowpass_freq, Q=lowpass_Q)
        y = highpass_biquad(y, sample_rate=self.sr, cutoff_freq=highpass_freq, Q=highpass_Q)
        return y.view(*x.shape)

    def parameter_size(self):
        return {
            "peak1_freq": 1,
            "peak1_gain": 1,
            "peak1_Q": 1,
            "peak2_freq": 1,
            "peak2_gain": 1,
            "peak2_Q": 1,
            "lowshelf_freq": 1,
            "lowshelf_gain": 1,
            "highshelf_freq": 1,
            "highshelf_gain": 1,
            "lowpass_freq": 1,
            "lowpass_Q": 1,
            "highpass_freq": 1,
            "highpass_Q": 1,
        }

    def decode_params(self, raw_params):
        return {
            "peak1_freq": self._minmax(raw_params["peak1_freq"], 33.0, 5400.0),
            "peak1_gain": raw_params["peak1_gain"],
            "peak1_Q": self._minmax(raw_params["peak1_Q"], 0.2, 20.0),
            "peak2_freq": self._minmax(raw_params["peak2_freq"], 200.0, 17500.0),
            "peak2_gain": raw_params["peak2_gain"],
            "peak2_Q": self._minmax(raw_params["peak2_Q"], 0.2, 20.0),
            "lowshelf_freq": self._minmax(raw_params["lowshelf_freq"], 30.0, 200.0),
            "lowshelf_gain": raw_params["lowshelf_gain"],
            "highshelf_freq": self._minmax(raw_params["highshelf_freq"], 750.0, 8300.0),
            "highshelf_gain": raw_params["highshelf_gain"],
            "lowpass_freq": self._minmax(raw_params["lowpass_freq"], 200.0, 18000.0),
            "lowpass_Q": self._minmax(raw_params["lowpass_Q"], 0.5, 10.0),
            "highpass_freq": self._minmax(raw_params["highpass_freq"], 16.0, 5300.0),
            "highpass_Q": self._minmax(raw_params["highpass_Q"], 0.5, 10.0),
        }


class Delay(FX):
    min_delay: float = 100
    max_delay: float = 1000

    def __init__(
        self,
        sr: int,
        delay=200.0,
        feedback=0.1,
        gain=0.1,
        ir_duration: float = 2,
        eq: Optional[nn.Module] = None,
        recursive_eq=False,
    ):
        super().__init__(
            delay=delay,
            feedback=feedback,
            gain=gain,
        )
        self.sr = sr
        self.ir_length = int(sr * max(ir_duration, self.max_delay * 0.002))

        register_parametrization(
            self.params, "delay", MinMax(self.min_delay, self.max_delay)
        )
        register_parametrization(self.params, "feedback", SmoothingCoef())
        register_parametrization(self.params, "gain", SmoothingCoef())

        self.eq = eq
        self.recursive_eq = recursive_eq

        self.register_buffer(
            "_arange", torch.arange(self.ir_length, dtype=torch.float32)
        )

        self.odd_pan = Panning(0)
        self.even_pan = Panning(0)

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        delay_in_samples = self.sr * self.params.delay * 0.001
        num_delays = self.ir_length // int(delay_in_samples.item() + 1)
        series = torch.arange(1, num_delays + 1, device=x.device)
        decays = self.params.feedback ** (series - 1)

        if self.recursive_eq and self.eq is not None:
            sinc_index = self._arange - delay_in_samples
            single_sinc_filter = torch.sinc(sinc_index)
            eq_sinc_filter = self.eq(single_sinc_filter)
            H = torch.fft.rfft(eq_sinc_filter)
            H_powered = torch.polar(
                H.abs() ** series.unsqueeze(-1), H.angle() * series.unsqueeze(-1)
            )
            sinc_filters = torch.fft.irfft(H_powered, n=self.ir_length)
        else:
            delays_in_samples = delay_in_samples * series
            sinc_indexes = self._arange - delays_in_samples.unsqueeze(-1)
            sinc_filters = torch.sinc(sinc_indexes)

        decayed_sinc_filters = sinc_filters * decays.unsqueeze(-1)
        return self._filter(x, decayed_sinc_filters)

    def _filter(self, x: torch.Tensor, decayed_sinc_filters: torch.Tensor):
        odd_delay_filters = torch.sum(decayed_sinc_filters[::2], 0)
        even_delay_filters = torch.sum(decayed_sinc_filters[1::2], 0)
        stacked_filters = torch.stack([odd_delay_filters, even_delay_filters])

        if self.eq is not None and not self.recursive_eq:
            stacked_filters = self.eq(stacked_filters)

        gained_odd_even_filters = stacked_filters * self.params.gain
        padded_x = F.pad(x, (gained_odd_even_filters.size(-1) - 1, 0))
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d
        return sum(
            [
                panner(s)
                for panner, s in zip(
                    [self.odd_pan, self.even_pan],
                    # fft_conv1d(
                    conv1d(
                        padded_x,
                        gained_odd_even_filters.flip(-1).unsqueeze(1),
                    ).chunk(2, 1),
                )
            ]
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"delay: {self.sr * self.params.delay.item() * 0.001} (samples)\n"
                f"feedback: {self.params.feedback.item()}\n"
                f"gain: {self.params.gain.item()}"
            )
        return s


class SurrogateDelay(Delay):
    def __init__(self, *args, dropout=0.5, straight_through=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = dropout
        self.straight_through = straight_through
        self.log_damp = nn.Parameter(torch.ones(1) * -0.01)
        register_parametrization(self, "log_damp", AlwaysNegative())

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        if not self.training:
            return super().forward(x)

        log_damp = self.log_damp
        delay_in_samples = self.sr * self.params.delay * 0.001
        num_delays = self.ir_length // int(delay_in_samples.item() + 1)
        series = torch.arange(1, num_delays + 1, device=x.device)
        decays = self.params.feedback ** (series - 1)

        if self.recursive_eq and self.eq is not None:
            exp_factor = self._arange[: self.ir_length // 2 + 1]
            damped_exp = torch.exp(
                log_damp * exp_factor
                - 1j * delay_in_samples / self.ir_length * 2 * torch.pi * exp_factor
            )
            sinc_filter = torch.fft.irfft(damped_exp, n=self.ir_length)
            if self.straight_through:
                sinc_index = self._arange - delay_in_samples
                hard_sinc_filter = torch.sinc(sinc_index)
                sinc_filter = sinc_filter + (hard_sinc_filter - sinc_filter).detach()

            eq_sinc_filter = self.eq(sinc_filter)
            H = torch.fft.rfft(eq_sinc_filter)

            # use polar form to avoid NaN
            H_powered = torch.polar(
                H.abs() ** series.unsqueeze(-1), H.angle() * series.unsqueeze(-1)
            )
            sinc_filters = torch.fft.irfft(H_powered, n=self.ir_length)
        else:
            exp_factors = series.unsqueeze(-1) * self._arange[: self.ir_length // 2 + 1]
            damped_exps = torch.exp(
                log_damp * exp_factors
                - 1j * delay_in_samples / self.ir_length * 2 * torch.pi * exp_factors
            )
            sinc_filters = torch.fft.irfft(damped_exps, n=self.ir_length)
            if self.straight_through:
                delays_in_samples = delay_in_samples * series
                sinc_indexes = self._arange - delays_in_samples.unsqueeze(-1)
                hard_sinc_filters = torch.sinc(sinc_indexes)
                sinc_filters = (
                    sinc_filters + (hard_sinc_filters - sinc_filters).detach()
                )

        decayed_sinc_filters = sinc_filters * decays.unsqueeze(-1)

        dropout_mask = torch.rand(x.size(0), device=x.device) < self.dropout
        if not torch.any(dropout_mask):
            return self._filter(x, decayed_sinc_filters)
        elif torch.all(dropout_mask):
            return super().forward(x)

        out = torch.zeros((x.size(0), 2, x.size(2)), device=x.device)
        out[~dropout_mask] = self._filter(x[~dropout_mask], decayed_sinc_filters)
        out[dropout_mask] = super().forward(x[dropout_mask])
        return out

    def extra_repr(self) -> str:
        with torch.no_grad():
            return super().extra_repr() + f"\ndamp: {self.log_damp.exp().item()}"


class FSDelay(FX):
    def __init__(
        self,
        sr: int,
        delay=200.0,
        feedback=0.1,
        gain=0.1,
        ir_duration: float = 6,
        eq: Optional[LowPass] = None,
        recursive_eq=False,
    ):
        super().__init__(
            delay=delay,
            feedback=feedback,
            gain=gain,
        )
        self.sr = sr
        self.ir_length = int(sr * max(ir_duration, Delay.max_delay * 0.002))

        register_parametrization(
            self.params, "delay", MinMax(Delay.min_delay, Delay.max_delay)
        )
        register_parametrization(self.params, "gain", SmoothingCoef())

        T_60 = ir_duration * 0.75
        max_delay_in_samples = sr * Delay.max_delay * 0.001
        maximum_decay = db2amp(torch.tensor(-60 / sr / T_60 * max_delay_in_samples))
        register_parametrization(self.params, "feedback", MinMax(0, maximum_decay))

        self.eq = eq
        self.recursive_eq = recursive_eq

        self.odd_pan = Panning(0)
        self.even_pan = Panning(0)

        self.register_buffer(
            "_arange", torch.arange(self.ir_length, dtype=torch.float32)
        )

    def _get_h(self):
        freqs = self._arange[: self.ir_length // 2 + 1] / self.ir_length * 2 * torch.pi
        delay_in_samples = self.sr * self.params.delay * 0.001

        # construct it like a fdn
        Dinv = torch.exp(1j * freqs * delay_in_samples)
        Dinv2 = torch.exp(2j * freqs * delay_in_samples)
        if self.recursive_eq and self.eq is not None:
            b0, b1, b2, a0, a1, a2 = module2coeffs(self.eq)
            z_inv = torch.exp(-1j * freqs)
            z_inv2 = torch.exp(-2j * freqs)
            eq_H = (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
            damp = eq_H * self.params.feedback
            det = Dinv2 - damp * damp
        else:
            damp = torch.full_like(Dinv, self.params.feedback) + 0j
            det = Dinv2 - self.params.feedback.square()
        inv_Dinv_m_A = torch.stack([Dinv, damp], 0) / det
        h = torch.fft.irfft(inv_Dinv_m_A, n=self.ir_length) * self.params.gain

        if self.eq is not None and not self.recursive_eq:
            h = self.eq(h)
        return h

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        h = self._get_h()

        padded_x = F.pad(x, (h.size(-1) - 1, 0))
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d
        return sum(
            [
                panner(s)
                for panner, s in zip(
                    [self.odd_pan, self.even_pan],
                    conv1d(
                        padded_x,
                        h.flip(-1).unsqueeze(1),
                    ).chunk(2, 1),
                )
            ]
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"delay: {self.sr * self.params.delay.item() * 0.001} (samples)\n"
                f"feedback: {self.params.feedback.item()}\n"
                f"gain: {self.params.gain.item()}"
            )
        return s


class FSSurrogateDelay(FSDelay):
    def __init__(self, *args, straight_through=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.straight_through = straight_through
        self.log_damp = nn.Parameter(torch.ones(1) * -0.0001)
        register_parametrization(self, "log_damp", AlwaysNegative())

    def _get_h(self):
        if not self.training:
            return super()._get_h()

        log_damp = self.log_damp
        delay_in_samples = self.sr * self.params.delay * 0.001

        exp_factor = self._arange[: self.ir_length // 2 + 1]
        freqs = exp_factor / self.ir_length * 2 * torch.pi
        D = torch.exp(log_damp * exp_factor - 1j * delay_in_samples * freqs)
        D2 = torch.exp(log_damp * exp_factor * 2 - 2j * delay_in_samples * freqs)

        if self.straight_through:
            D_orig = torch.exp(-1j * delay_in_samples * freqs)
            D2_orig = torch.exp(-2j * delay_in_samples * freqs)
            D = torch.stack([D, D_orig], 0)
            D2 = torch.stack([D2, D2_orig], 0)

        if self.recursive_eq and self.eq is not None:
            b0, b1, b2, a0, a1, a2 = module2coeffs(self.eq)
            z_inv = torch.exp(-1j * freqs)
            z_inv2 = torch.exp(-2j * freqs)
            eq_H = (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
            damp = eq_H * self.params.feedback
            odd_H = D / (1 - damp * damp * D2)
            even_H = odd_H * D * damp
        else:
            damp = torch.full_like(D, self.params.feedback) + 0j
            odd_H = D / (1 - self.params.feedback.square() * D2)
            even_H = odd_H * D * self.params.feedback

        inv_Dinv_m_A = torch.stack([odd_H, even_H], 0)
        h = torch.fft.irfft(inv_Dinv_m_A, n=self.ir_length)

        if self.straight_through:
            damped_h, orig_h = h.unbind(1)
            h = damped_h + (orig_h - damped_h).detach()

        if self.eq is not None and not self.recursive_eq:
            h = self.eq(h)
        return h * self.params.gain

    def extra_repr(self) -> str:
        with torch.no_grad():
            return super().extra_repr() + f"\ndamp: {self.log_damp.exp().item()}"


class FSDelay_diffvox(FX):
    min_delay_ms: float = 100.0
    max_delay_ms: float = 1000.0

    def __init__(
        self,
        sample_rate: int = 48000,
        delay: float = 400.0,
        feedback: float = 0.1,
        gain: float = 0.1,
        ir_duration: float =2.0, # 3.7s to match our frame size. but to match paper, it needs to be one third of audio_len, thus 1.2s. 2s to match grafx's delay and reverb
        recursive_eq: bool = True,
        eq_freq: float = 8000.0,
        eq_Q: float = 0.707,
        eq_min_freq: float = 200.0,
        eq_max_freq: float = 16000.0,
        eq_min_Q: float = 0.5,
        eq_max_Q: float = 1.0,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
    ):
        super().__init__(
            delay=delay,
            feedback=feedback,
            gain=gain,
            eq_freq=eq_freq,
            eq_Q=eq_Q,
        )
        self.sr = sample_rate
        self.ir_duration = ir_duration
        self.ir_length = int(sample_rate * max(ir_duration, self.max_delay_ms * 0.002))
        self.recursive_eq = recursive_eq
        self.eq_min_freq = eq_min_freq
        self.eq_max_freq = eq_max_freq
        self.eq_min_Q = eq_min_Q
        self.eq_max_Q = eq_max_Q

        T_60 = ir_duration * 0.75
        max_delay_in_samples = sample_rate * self.max_delay_ms * 0.001
        maximum_decay = db2amp(torch.tensor(-60 / sample_rate / T_60 * max_delay_in_samples))
        self.register_buffer("_maximum_decay", maximum_decay.float(), persistent=False)
        self.register_buffer("_arange", torch.arange(self.ir_length, dtype=torch.float32), persistent=False)

    def _smoothing_coef(self, x):
        return x.sigmoid()

    def _minmax(self, x, lo: float, hi: float):
        return lo + (hi - lo) * x.sigmoid()

    def _node_param(self, p, N):
        return p.view(N, -1)[:, 0]

    def _downmix_to_mono(self, x):
        if x.size(1) == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    def _compute_h(self, delay_ms, feedback_amp, gain_amp, eq_freq_hz, eq_q):
        B = delay_ms.shape[0]
        device = delay_ms.device
        freqs = self._arange[: self.ir_length // 2 + 1].to(device=device, dtype=delay_ms.dtype)
        freqs = freqs / self.ir_length * 2.0 * torch.pi
        delay_in_samples = self.sr * delay_ms * 0.001

        Dinv = torch.exp(1j * freqs[None, :] * delay_in_samples[:, None])
        Dinv2 = torch.exp(2j * freqs[None, :] * delay_in_samples[:, None])

        if self.recursive_eq:
            b0, b1, b2, a0, a1, a2 = lowpass_biquad_coef(
                sample_rate=self.sr,
                cutoff_freq=eq_freq_hz,
                Q=eq_q,
            )
            z_inv = torch.exp(-1j * freqs)[None, :]
            z_inv2 = torch.exp(-2j * freqs)[None, :]
            eq_H = (
                b0[:, None] + b1[:, None] * z_inv + b2[:, None] * z_inv2
            ) / (
                a0[:, None] + a1[:, None] * z_inv + a2[:, None] * z_inv2
            )
            damp = eq_H * feedback_amp[:, None]
            det = Dinv2 - damp * damp
        else:
            damp = feedback_amp[:, None] + 0j
            det = Dinv2 - feedback_amp[:, None].square()

        inv_Dinv_m_A = torch.stack([Dinv, damp], dim=1) / det[:, None, :]
        h = torch.fft.irfft(inv_Dinv_m_A, n=self.ir_length, dim=-1) * gain_amp[:, None, None]

        if not self.recursive_eq:
            h_flat = h.reshape(B * 2, self.ir_length)
            eq_freq_rep = eq_freq_hz.repeat_interleave(2)
            eq_q_rep = eq_q.repeat_interleave(2)
            h_flat = lowpass_biquad(
                h_flat, sample_rate=self.sr, cutoff_freq=eq_freq_rep, Q=eq_q_rep
            )
            h = h_flat.view(B, 2, self.ir_length)
        return h

    def _stereo_convolve(self, x_mono, h):
        B, _, T = x_mono.shape
        x_flat = x_mono[:, 0, :]
        padded = F.pad(x_flat, (h.size(-1) - 1, 0))
        conv1d = F.conv1d if T > 44100 * 20 else fft_conv1d
        y_pair = conv1d(
            padded.unsqueeze(0),
            h.flip(-1).reshape(B * 2, 1, h.size(-1)),
            groups=B,
        ).view(B, 2, T)
        y_sum = y_pair[:, 0:1, :] + y_pair[:, 1:2, :]
        return y_sum.expand(B, 2, T)

    def forward(
        self,
        x,
        delay,
        feedback,
        gain,
        eq_freq,
        eq_Q,
    ):
        N, _, _ = x.shape
        x_mono = self._downmix_to_mono(x)

        delay_ms = self._minmax(self._node_param(delay, N), self.min_delay_ms, self.max_delay_ms)
        feedback_amp = self._smoothing_coef(self._node_param(feedback, N)) * self._maximum_decay.to(delay_ms)
        gain_amp = self._smoothing_coef(self._node_param(gain, N))
        eq_freq_hz = self._minmax(self._node_param(eq_freq, N), self.eq_min_freq, self.eq_max_freq)
        eq_q = self._minmax(self._node_param(eq_Q, N), self.eq_min_Q, self.eq_max_Q)

        h = self._compute_h(delay_ms, feedback_amp, gain_amp, eq_freq_hz, eq_q)
        y = self._stereo_convolve(x_mono, h)
        return y, {"radii_reg": torch.zeros((), device=x.device, dtype=x.dtype)}

    def parameter_size(self):
        return {
            "delay": 1,
            "feedback": 1,
            "gain": 1,
            "eq_freq": 1,
            "eq_Q": 1,
        }

    def decode_params(self, raw_params):
        probe = next(iter(raw_params.values()))
        maximum_decay = self._maximum_decay.to(device=probe.device, dtype=probe.dtype)
        return {
            "delay": self._minmax(raw_params["delay"], self.min_delay_ms, self.max_delay_ms),
            "feedback": self._smoothing_coef(raw_params["feedback"]) * maximum_decay,
            "gain": self._smoothing_coef(raw_params["gain"]),
            "eq_freq": self._minmax(raw_params["eq_freq"], self.eq_min_freq, self.eq_max_freq),
            "eq_Q": self._minmax(raw_params["eq_Q"], self.eq_min_Q, self.eq_max_Q),
        }


class FSSurrogateDelay_diffvox(FSDelay_diffvox):
    def __init__(self, *args, straight_through: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.straight_through = straight_through

    def _always_negative(self, x):
        return -F.softplus(x)

    def _compute_h(self, delay_ms, feedback_amp, gain_amp, eq_freq_hz, eq_q, log_damp_raw=None):
        if (not self.training) or (log_damp_raw is None):
            return super()._compute_h(delay_ms, feedback_amp, gain_amp, eq_freq_hz, eq_q)

        device = delay_ms.device
        freqs = self._arange[: self.ir_length // 2 + 1].to(device=device, dtype=delay_ms.dtype)
        exp_factor = freqs
        freqs = freqs / self.ir_length * 2.0 * torch.pi

        delay_in_samples = self.sr * delay_ms * 0.001
        log_damp = self._always_negative(log_damp_raw)

        D = torch.exp(log_damp[:, None] * exp_factor[None, :] - 1j * delay_in_samples[:, None] * freqs[None, :])
        D2 = torch.exp(log_damp[:, None] * exp_factor[None, :] * 2.0 - 2j * delay_in_samples[:, None] * freqs[None, :])

        if self.straight_through:
            D_orig = torch.exp(-1j * delay_in_samples[:, None] * freqs[None, :])
            D2_orig = torch.exp(-2j * delay_in_samples[:, None] * freqs[None, :])
            D = D + (D_orig - D).detach()
            D2 = D2 + (D2_orig - D2).detach()

        if self.recursive_eq:
            b0, b1, b2, a0, a1, a2 = lowpass_biquad_coef(
                sample_rate=self.sr,
                cutoff_freq=eq_freq_hz,
                Q=eq_q,
            )
            z_inv = torch.exp(-1j * freqs)[None, :]
            z_inv2 = torch.exp(-2j * freqs)[None, :]
            eq_H = (
                b0[:, None] + b1[:, None] * z_inv + b2[:, None] * z_inv2
            ) / (
                a0[:, None] + a1[:, None] * z_inv + a2[:, None] * z_inv2
            )
            damp = eq_H * feedback_amp[:, None]
            odd_H = D / (1.0 - damp * damp * D2)
            even_H = odd_H * D * damp
        else:
            odd_H = D / (1.0 - feedback_amp[:, None].square() * D2)
            even_H = odd_H * D * feedback_amp[:, None]

        h = torch.fft.irfft(torch.stack([odd_H, even_H], dim=1), n=self.ir_length, dim=-1)

        if not self.recursive_eq:
            B = delay_ms.shape[0]
            h_flat = h.reshape(B * 2, self.ir_length)
            eq_freq_rep = eq_freq_hz.repeat_interleave(2)
            eq_q_rep = eq_q.repeat_interleave(2)
            h_flat = lowpass_biquad(
                h_flat, sample_rate=self.sr, cutoff_freq=eq_freq_rep, Q=eq_q_rep
            )
            h = h_flat.view(B, 2, self.ir_length)
        return h * gain_amp[:, None, None]

    def forward(
        self,
        x,
        delay,
        feedback,
        gain,
        eq_freq,
        eq_Q,
        log_damp,
    ):
        N, _, _ = x.shape
        x_mono = self._downmix_to_mono(x)

        delay_ms = self._minmax(self._node_param(delay, N), self.min_delay_ms, self.max_delay_ms)
        feedback_amp = self._smoothing_coef(self._node_param(feedback, N)) * self._maximum_decay.to(delay_ms)
        gain_amp = self._smoothing_coef(self._node_param(gain, N))
        eq_freq_hz = self._minmax(self._node_param(eq_freq, N), self.eq_min_freq, self.eq_max_freq)
        eq_q = self._minmax(self._node_param(eq_Q, N), self.eq_min_Q, self.eq_max_Q)
        log_damp_raw = self._node_param(log_damp, N)

        h = self._compute_h(
            delay_ms=delay_ms,
            feedback_amp=feedback_amp,
            gain_amp=gain_amp,
            eq_freq_hz=eq_freq_hz,
            eq_q=eq_q,
            log_damp_raw=log_damp_raw,
        )
        y = self._stereo_convolve(x_mono, h)
        return y, {"radii_reg": torch.zeros((), device=x.device, dtype=x.dtype)}

    def parameter_size(self):
        size = super().parameter_size()
        size["log_damp"] = 1
        return size

    def decode_params(self, raw_params):
        decoded = super().decode_params(raw_params)
        decoded["log_damp"] = -F.softplus(raw_params["log_damp"])
        return decoded


class SendFXsAndSum(FX):
    def __init__(self, *args, cross_send=True, pan_direct=False):
        super().__init__(
            **(
                {
                    f"sends_{i}": torch.full([len(args) - i - 1], 0.01)
                    for i in range(len(args) - 1)
                }
                if cross_send
                else {}
            )
        )
        self.effects = nn.ModuleList(args)
        if pan_direct:
            self.pan = Panning()

        if cross_send:
            for i in range(len(args) - 1):
                register_parametrization(self.params, f"sends_{i}", SmoothingCoef())

    def forward(self, x):
        if hasattr(self, "pan"):
            di = self.pan(x)
        else:
            di = x

        if len(self.params) == 0:
            return reduce(
                lambda x, y: x[..., : y.shape[-1]] + y[..., : x.shape[-1]],
                map(lambda f: f(x), self.effects),
                di,
            )

        def f(states, ps):
            x, cum_sends = states
            m, send_gains = ps
            h = m(cum_sends[0])
            return (
                x[..., : h.shape[-1]] + h[..., : x.shape[-1]],
                (
                    None
                    if cum_sends.size(0) == 1
                    else cum_sends[1:, ..., : h.shape[-1]]
                    + send_gains[:, None, None, None] * h[..., : cum_sends.shape[-1]]
                ),
            )

        return reduce(
            f,
            zip(
                self.effects,
                [self.params[f"sends_{i}"] for i in range(len(self.effects) - 1)]
                + [None],
            ),
            (di, x.unsqueeze(0).expand(len(self.effects), -1, -1, -1)),
        )[0]


class UniLossLess(nn.Module):
    def forward(self, x):
        tri = x.triu(1)
        return torch.linalg.matrix_exp(tri - tri.T)


class FDN(FX):
    max_delay = 100

    def __init__(
        self,
        sr: int,
        ir_duration: float = 1.0,
        delays=(997, 1153, 1327, 1559, 1801, 2099),
        trainable_delay=False,
        num_decay_freq=1,
        delay_independent_decay=False,
        eq: Optional[nn.Module] = None,
    ):
        # beta = torch.distributions.Beta(1.1, 6)
        num_delays = len(delays)
        super().__init__(
            b=torch.ones(num_delays, 2) / num_delays,
            c=torch.zeros(2, num_delays),
            U=torch.randn(num_delays, num_delays) / num_delays**0.5,
            gamma=torch.rand(
                num_decay_freq, num_delays if not delay_independent_decay else 1
            )
            * 0.2
            + 0.4,
            # delays=beta.sample((num_delays,)) * 64,
        )

        self.sr = sr
        self.ir_length = int(sr * ir_duration)

        # ir_duration = T_60
        T_60 = ir_duration * 0.75
        delays = torch.tensor(delays)
        if delay_independent_decay:
            gamma_max = db2amp(-60 / sr / T_60 * delays.min())
        else:
            gamma_max = db2amp(-60 / sr / T_60 * delays)

        register_parametrization(self.params, "gamma", MinMax(0, gamma_max))
        register_parametrization(self.params, "U", UniLossLess())

        if not trainable_delay:
            self.register_buffer(
                "delays",
                delays,
            )
        else:
            self.params["delays"] = nn.Parameter(delays / sr * 1000)
            register_parametrization(self.params, "delays", MinMax(0, self.max_delay))

        self.register_forward_pre_hook(broadcast2stereo)

        self.eq = eq

    def forward(self, x):
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d

        c = self.params.c + 0j
        b = self.params.b + 0j

        gamma = self.params.gamma
        delays = self.delays if hasattr(self, "delays") else self.params.delays

        if gamma.size(0) > 1:
            gamma = F.interpolate(
                gamma.T.unsqueeze(1),
                size=self.ir_length // 2 + 1,
                align_corners=True,
                mode="linear",
            ).transpose(0, 2)

        if gamma.size(2) == 1:
            gamma = gamma ** (delays / delays.min())

        A = self.params.U * gamma

        freqs = (
            torch.arange(self.ir_length // 2 + 1, device=x.device)
            / self.ir_length
            * 2
            * torch.pi
        )
        invD = torch.exp(1j * freqs[:, None] * delays)
        # H = c @ torch.linalg.inv(torch.diag_embed(invD) - A) @ b
        H = c @ torch.linalg.solve(torch.diag_embed(invD) - A, b)

        h = torch.fft.irfft(H.permute(1, 2, 0), n=self.ir_length)

        if self.eq is not None:
            h = self.eq(h)

        # return fft_conv1d(
        return conv1d(
            F.pad(x, (self.ir_length - 1, 0)),
            h.flip(-1),
        )



# MODIFIED TO FIT GRAFX



class CompressorExpander_diffvox(FX):
    cmp_ratio_min: float = 1
    cmp_ratio_max: float = 20

    def __init__(
        self,
        sample_rate: int = 30000,
        cmp_ratio: float = 2.0,
        exp_ratio: float = 0.5,
        at_ms: float = 50.0,
        rt_ms: float = 50.0,
        avg_coef: float = 0.3,
        cmp_th: float = -18.0,
        exp_th: float = -54.0,
        make_up: float = 0.0,
        delay: int = 0,
        lookahead: bool = False,
        max_lookahead: float = 15.0,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
    ):
        super().__init__(
            cmp_th=cmp_th,
            exp_th=exp_th,
            make_up=make_up,
            avg_coef=avg_coef,
            cmp_ratio=cmp_ratio,
            exp_ratio=exp_ratio,
        )
        # deprecated, please use lookahead instead
        self.delay = delay
        sr = sample_rate
        self.sr = sr
        self.max_lookahead = max_lookahead
        self._lookahead_cache_key = None
        self._lookahead_cache_values = None
        self._lookahead_cache_filters = None
        self._lookahead_cache_shared = False

        self.params["at"] = nn.Parameter(ms2coef(torch.tensor(at_ms), sr))
        self.params["rt"] = nn.Parameter(ms2coef(torch.tensor(rt_ms), sr))

        if lookahead:
            self.params["lookahead"] = nn.Parameter(torch.ones(1) / sr * 1000)
            register_parametrization(
                self.params, "lookahead", WrappedPositive(max_lookahead)
            )
            sinc_length = int(sr * (max_lookahead + 1) * 0.001) + 1
            left_pad_size = int(sr * 0.001)
            self._pad_size = (left_pad_size, sinc_length - left_pad_size - 1)
            self.register_buffer(
                "_arange",
                torch.arange(sinc_length) - left_pad_size,
                persistent=False,
            )
        self.lookahead = lookahead

        register_parametrization(self.params, "at", SmoothingCoef())
        register_parametrization(self.params, "rt", SmoothingCoef())
        register_parametrization(self.params, "avg_coef", SmoothingCoef())
        register_parametrization(
            self.params, "cmp_ratio", MinMax(self.cmp_ratio_min, self.cmp_ratio_max)
        )
        register_parametrization(self.params, "exp_ratio", SmoothingCoef())

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"attack: {coef2ms(self.params.at, self.sr).item()} (ms)\n"
                f"release: {coef2ms(self.params.rt, self.sr).item()} (ms)\n"
                f"avg_coef: {self.params.avg_coef.item()}\n"
                f"compressor_ratio: {self.params.cmp_ratio.item()}\n"
                f"expander_ratio: {self.params.exp_ratio.item()}\n"
                f"compressor_threshold: {self.params.cmp_th.item()} (dB)\n"
                f"expander_threshold: {self.params.exp_th.item()} (dB)\n"
                f"make_up: {self.params.make_up.item()} (dB)"
            )
            if self.lookahead:
                s += f"\nlookahead: {self.params.lookahead.item()} (ms)"
        return s

    def _smoothing_coef(self, x):
        return x.sigmoid()

    def _wrapped_positive(self, x, period: float):
        return x.abs() % period

    def _minmax(self, x, lo: float, hi: float):
        return lo + (hi - lo) * x.sigmoid()

    def _node_param(self, p: torch.Tensor, n_nodes: int):
        return p.view(n_nodes, -1)[:, 0]

    def _expand_node_params(self, p_node: torch.Tensor, channels: int):
        # Expand once to [N, C] then flatten to [N*C] to avoid repeated interleave ops.
        return p_node[:, None].expand(-1, channels).reshape(-1)

    def _lookahead_filters(
        self,
        lookahead_ms_nc: torch.Tensor,
        n_nodes: int,
        channels: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        cache_key = (
            device.type,
            device.index,
            str(dtype),
            int(n_nodes),
            int(channels),
            int(self._arange.numel()),
        )
        can_reuse_cache = (
            (not torch.is_grad_enabled())
            or (not lookahead_ms_nc.requires_grad)
        )
        if (
            can_reuse_cache
            and self._lookahead_cache_key == cache_key
            and self._lookahead_cache_values is not None
            and self._lookahead_cache_filters is not None
            and torch.equal(self._lookahead_cache_values, lookahead_ms_nc)
        ):
            return self._lookahead_cache_filters, self._lookahead_cache_shared

        arange = self._arange.to(device=device, dtype=dtype)
        lookahead_in_samples = lookahead_ms_nc * 0.001 * self.sr
        shared = bool(
            torch.max((lookahead_ms_nc - lookahead_ms_nc[:1]).abs()) <= 1e-6
        )

        if shared:
            filters = torch.sinc(arange - lookahead_in_samples[0])
        else:
            filters = torch.sinc(arange.unsqueeze(0) - lookahead_in_samples.unsqueeze(-1))

        if can_reuse_cache:
            self._lookahead_cache_key = cache_key
            self._lookahead_cache_values = lookahead_ms_nc.detach().clone()
            self._lookahead_cache_filters = filters
            self._lookahead_cache_shared = shared

        return filters, shared

    def forward(self, 
                x,
                at,
                rt,
                avg_coef,
                cmp_ratio,
                exp_ratio,
                cmp_th,
                exp_th,
                make_up,
                lookahead=None,):

        N, C, T = x.shape
        x_flat = x.contiguous().flatten(0, 1)
        
        at_coef = self._node_param(self._smoothing_coef(at), N)
        rt_coef = self._node_param(self._smoothing_coef(rt), N)
        avg_c = self._node_param(self._smoothing_coef(avg_coef), N)
        exp_r = self._node_param(self._smoothing_coef(exp_ratio), N)
        cmp_r = self._node_param(
            self._minmax(cmp_ratio, self.cmp_ratio_min, self.cmp_ratio_max), N
        )
        cmp_th_n = self._node_param(cmp_th, N)
        exp_th_n = self._node_param(exp_th, N)
        make_up_n = self._node_param(make_up, N)

        packed_n = torch.stack(
            [at_coef, rt_coef, avg_c, cmp_r, exp_r, cmp_th_n, exp_th_n, make_up_n], dim=-1
        )
        packed_nc = packed_n[:, None, :].expand(-1, C, -1).reshape(N * C, 8)
        at_coef, rt_coef, avg_c, cmp_r, exp_r, cmp_th, exp_th, make_up = packed_nc.unbind(dim=-1)
                
        if self.lookahead:
            lookahead_ms_n = self._node_param(
                self._wrapped_positive(lookahead, self.max_lookahead), N
            )
            lookahead_ms_nc = self._expand_node_params(lookahead_ms_n, C)
            sinc_filters, shared_lookahead = self._lookahead_filters(
                lookahead_ms_nc=lookahead_ms_nc,
                n_nodes=N,
                channels=C,
                dtype=x.dtype,
                device=x.device,
            )

            def lookahead_func(gain):
                gain_3d = gain.view(-1, 1, gain.size(-1))
                gain_pad = F.pad(gain_3d, self._pad_size, mode="replicate")
                if shared_lookahead:
                    # Fast path: one shared kernel across all streams.
                    return F.conv1d(gain_pad, sinc_filters[None, None, :]).view(*gain.shape)
                gain_out = F.conv1d(
                    gain_pad.transpose(0, 1),
                    sinc_filters.unsqueeze(1),
                    groups=gain.size(0),
                )
                return gain_out.view(*gain.shape)
        else:
            lookahead_func = lambda g: g


        y = compressor_expander(
            x_flat,
            lookahead_func=lookahead_func,
            at=at_coef,
            rt=rt_coef,
            avg_coef=avg_c,
            cmp_ratio=cmp_r,
            exp_ratio=exp_r,
            cmp_th=cmp_th,
            exp_th=exp_th,
            make_up=make_up,
            ).view(*x.shape)
        
        return y

    def parameter_size(self):
        size = {}
        size["at"] = 1
        size["rt"] = 1
        size["avg_coef"] = 1
        size["cmp_ratio"] = 1
        size["exp_ratio"] = 1
        size["cmp_th"] = 1
        size["exp_th"] = 1
        size["make_up"] = 1
        if self.lookahead:
            size["lookahead"] = 1
        return size

    def decode_params(self, raw_params):
        decoded = {
            "at": self._smoothing_coef(raw_params["at"]),
            "rt": self._smoothing_coef(raw_params["rt"]),
            "avg_coef": self._smoothing_coef(raw_params["avg_coef"]),
            "cmp_ratio": self._minmax(raw_params["cmp_ratio"], self.cmp_ratio_min, self.cmp_ratio_max),
            "exp_ratio": self._smoothing_coef(raw_params["exp_ratio"]),
            "cmp_th": raw_params["cmp_th"],
            "exp_th": raw_params["exp_th"],
            "make_up": raw_params["make_up"],
        }
        if self.lookahead and "lookahead" in raw_params:
            decoded["lookahead"] = self._wrapped_positive(raw_params["lookahead"], self.max_lookahead)
        return decoded


class FDN_diffvox(FX):
    def __init__(
        self,
        sample_rate: int = 48000,
        ir_duration: float = 2.0,
        delays=(997, 1153, 1327, 1559, 1801, 2099),
        num_decay_freq: int = 49,
        delay_independent_decay: bool = True,
        eq_peak1_freq: float = 800.0,
        eq_peak1_gain: float = 0.0,
        eq_peak1_Q: float = 0.707,
        eq_peak2_freq: float = 4000.0,
        eq_peak2_gain: float = 0.0,
        eq_peak2_Q: float = 0.707,
        eq_lowshelf_freq: float = 115.0,
        eq_lowshelf_gain: float = 0.0,
        eq_highshelf_freq: float = 8000.0,
        eq_highshelf_gain: float = 0.0,
        flashfftconv: bool = False,
        max_input_len: int = 2**17,
    ):
        n_delays = len(delays)
        super().__init__(
            b=torch.ones(n_delays, 2) / n_delays,
            c=torch.zeros(2, n_delays),
            U=torch.randn(n_delays, n_delays) / (n_delays ** 0.5),
            gamma=torch.rand(num_decay_freq, 1 if delay_independent_decay else n_delays) * 0.2 + 0.4,
            eq_peak1_freq=eq_peak1_freq,
            eq_peak1_gain=eq_peak1_gain,
            eq_peak1_Q=eq_peak1_Q,
            eq_peak2_freq=eq_peak2_freq,
            eq_peak2_gain=eq_peak2_gain,
            eq_peak2_Q=eq_peak2_Q,
            eq_lowshelf_freq=eq_lowshelf_freq,
            eq_lowshelf_gain=eq_lowshelf_gain,
            eq_highshelf_freq=eq_highshelf_freq,
            eq_highshelf_gain=eq_highshelf_gain,
        )
        self.sr = int(sample_rate)
        self.ir_duration = float(ir_duration)
        self.ir_length = int(self.sr * self.ir_duration)
        self.num_decay_freq = int(num_decay_freq)
        self.delay_independent_decay = bool(delay_independent_decay)
        self.register_buffer("delays", torch.tensor(delays, dtype=torch.float32), persistent=False)

        T_60 = self.ir_duration * 0.75
        gamma_max = db2amp(
            -60.0 / self.sr / T_60 * (self.delays.min() if self.delay_independent_decay else self.delays)
        )
        if not torch.is_tensor(gamma_max):
            gamma_max = torch.tensor(gamma_max, dtype=torch.float32)
        self.register_buffer("_gamma_max", gamma_max.float(), persistent=False)

        self.eq_peak1_min_freq = 200.0
        self.eq_peak1_max_freq = 2500.0
        self.eq_peak1_min_q = 0.1
        self.eq_peak1_max_q = 3.0

        self.eq_peak2_min_freq = 600.0
        self.eq_peak2_max_freq = 7000.0
        self.eq_peak2_min_q = 0.1
        self.eq_peak2_max_q = 3.0

        self.eq_lowshelf_min_freq = 30.0
        self.eq_lowshelf_max_freq = 450.0
        self.eq_highshelf_min_freq = 1500.0
        self.eq_highshelf_max_freq = 16000.0
        self.register_buffer("_shelf_q", torch.tensor(0.707), persistent=False)
        self._cache_key = None
        self._cached_delays = None
        self._cached_freqs = None
        self._cached_invD = None
        self._cached_diag_invD = None

    def _sigmoid(self, x):
        return torch.sigmoid(x)

    def _minmax(self, x, lo: float, hi: float):
        return lo + (hi - lo) * torch.sigmoid(x)

    def _node_param(self, p, n_nodes: int):
        p = p.view(n_nodes, -1)
        return p[:, 0]

    def _orthogonal_from_raw(self, u_raw):
        tri = torch.triu(u_raw, diagonal=1)
        return torch.linalg.matrix_exp(tri - tri.transpose(-1, -2))

    def _activated_gamma(self, gamma_raw, n_delays: int):
        if gamma_raw.dim() == 1:
            gamma_raw = gamma_raw.view(1, 1)
        g = self._sigmoid(gamma_raw)
        gm = self._gamma_max.to(device=g.device, dtype=g.dtype)
        if gm.dim() == 0:
            g = g * gm
        else:
            view_shape = [1] * g.dim()
            view_shape[-1] = -1
            g = g * gm.view(*view_shape)
        return g

    def _build_A(self, gamma, delays, U):
        n_delays = int(delays.numel())
        num_freq_bins = self.ir_length // 2 + 1

        if gamma.dim() == 2:
            if gamma.size(1) in (1, n_delays):
                gamma = gamma.unsqueeze(1)
            else:
                gamma = F.interpolate(
                    gamma.unsqueeze(1),
                    size=num_freq_bins,
                    align_corners=True,
                    mode="linear",
                ).transpose(1, 2)
        elif gamma.dim() == 3:
            # [N, F, D] or [N, F, 1]
            gamma = F.interpolate(
                gamma.transpose(1, 2),
                size=num_freq_bins,
                align_corners=True,
                mode="linear",
            ).transpose(1, 2)
        else:
            raise ValueError(f"Unexpected gamma shape: {tuple(gamma.shape)}")

        if gamma.dim() == 3 and gamma.size(-1) == 1:
            gamma = gamma ** (delays / delays.min()).view(1, 1, -1)
        elif gamma.dim() == 3 and gamma.size(-1) != n_delays:
            if gamma.size(-1) > n_delays:
                gamma = gamma[..., :n_delays]
            else:
                pad = gamma[..., -1:].expand(*gamma.shape[:-1], n_delays - gamma.size(-1))
                gamma = torch.cat([gamma, pad], dim=-1)
        return U.unsqueeze(1) * gamma.unsqueeze(2)

    def _get_cached_terms(self, device: torch.device, dtype: torch.dtype):
        cache_key = (device.type, device.index, str(dtype), self.ir_length, int(self.delays.numel()))
        if self._cache_key != cache_key:
            delays = self.delays.to(device=device, dtype=dtype)
            freqs = (
                torch.arange(self.ir_length // 2 + 1, device=device, dtype=dtype)
                / self.ir_length
                * 2.0
                * torch.pi
            )
            ctype = torch.complex64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128
            invD = torch.exp(1j * freqs[:, None] * delays[None, :]).to(ctype)
            self._cache_key = cache_key
            self._cached_delays = delays
            self._cached_freqs = freqs
            self._cached_invD = invD
            self._cached_diag_invD = torch.diag_embed(invD)
        return self._cached_delays, self._cached_freqs, self._cached_invD, self._cached_diag_invD

    def _apply_internal_eq(
        self,
        h,
        peak1_freq,
        peak1_gain,
        peak1_q,
        peak2_freq,
        peak2_gain,
        peak2_q,
        lowshelf_freq,
        lowshelf_gain,
        highshelf_freq,
        highshelf_gain,
    ):
        n_nodes = h.shape[0]
        n_filters = int(h.shape[0] * h.shape[1] * h.shape[2])
        h_flat = h.reshape(n_filters, h.shape[-1])

        device = h.device
        dtype = h.dtype
        shelf_q = self._shelf_q.to(device=device, dtype=dtype).expand(n_filters)
        repeat_per_node = int(h.shape[1] * h.shape[2])
        p1f = peak1_freq.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        p1g = peak1_gain.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        p1q = peak1_q.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        p2f = peak2_freq.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        p2g = peak2_gain.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        p2q = peak2_q.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        lsf = lowshelf_freq.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        lsg = lowshelf_gain.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        hsf = highshelf_freq.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)
        hsg = highshelf_gain.reshape(n_nodes).to(device=device, dtype=dtype).repeat_interleave(repeat_per_node)

        h_flat = equalizer_biquad(
            h_flat, sample_rate=self.sr, center_freq=p1f, Q=p1q, gain=p1g
        )
        h_flat = equalizer_biquad(
            h_flat, sample_rate=self.sr, center_freq=p2f, Q=p2q, gain=p2g
        )
        h_flat = lowshelf_biquad(
            h_flat, sample_rate=self.sr, cutoff_freq=lsf, gain=lsg, Q=shelf_q
        )
        h_flat = highshelf_biquad(
            h_flat, sample_rate=self.sr, cutoff_freq=hsf, gain=hsg, Q=shelf_q
        )
        return h_flat.view_as(h)

    def forward(
        self,
        x,
        b,
        c,
        U,
        gamma,
        eq_peak1_freq,
        eq_peak1_gain,
        eq_peak1_Q,
        eq_peak2_freq,
        eq_peak2_gain,
        eq_peak2_Q,
        eq_lowshelf_freq,
        eq_lowshelf_gain,
        eq_highshelf_freq,
        eq_highshelf_gain,
    ):
        n_nodes, in_ch, time_len = x.shape
        conv1d = F.conv1d if time_len > 44100 * 20 else fft_conv1d

        delays, _, _, diag_invD = self._get_cached_terms(device=x.device, dtype=x.dtype)
        n_delays = int(delays.numel())

        p1f = self._minmax(self._node_param(eq_peak1_freq, n_nodes), self.eq_peak1_min_freq, self.eq_peak1_max_freq)
        p1g = self._node_param(eq_peak1_gain, n_nodes)
        p1q = self._minmax(self._node_param(eq_peak1_Q, n_nodes), self.eq_peak1_min_q, self.eq_peak1_max_q)
        p2f = self._minmax(self._node_param(eq_peak2_freq, n_nodes), self.eq_peak2_min_freq, self.eq_peak2_max_freq)
        p2g = self._node_param(eq_peak2_gain, n_nodes)
        p2q = self._minmax(self._node_param(eq_peak2_Q, n_nodes), self.eq_peak2_min_q, self.eq_peak2_max_q)
        lsf = self._minmax(self._node_param(eq_lowshelf_freq, n_nodes), self.eq_lowshelf_min_freq, self.eq_lowshelf_max_freq)
        lsg = self._node_param(eq_lowshelf_gain, n_nodes)
        hsf = self._minmax(self._node_param(eq_highshelf_freq, n_nodes), self.eq_highshelf_min_freq, self.eq_highshelf_max_freq)
        hsg = self._node_param(eq_highshelf_gain, n_nodes)

        b_all = b.reshape(n_nodes, n_delays, 2).to(dtype=x.dtype)
        c_all = c.reshape(n_nodes, 2, n_delays).to(dtype=x.dtype)
        U_all = self._orthogonal_from_raw(U.reshape(n_nodes, n_delays, n_delays).to(dtype=x.dtype))
        gamma_all = self._activated_gamma(gamma, n_delays).to(dtype=x.dtype)
        A_all = self._build_A(gamma_all, delays, U_all)

        ctype = torch.complex64 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128
        lhs = diag_invD.unsqueeze(0) - A_all.to(ctype)
        rhs = b_all.to(ctype).unsqueeze(1).expand(-1, lhs.shape[1], -1, -1)
        solved = torch.linalg.solve(lhs, rhs)  # [N, F, D, 2]
        H = torch.einsum("nid,nfdk->nfik", c_all.to(ctype), solved)  # [N, F, 2, 2]
        h = torch.fft.irfft(H.permute(0, 2, 3, 1), n=self.ir_length, dim=-1)  # [N, 2, 2, L]
        h = self._apply_internal_eq(
            h,
            peak1_freq=p1f,
            peak1_gain=p1g,
            peak1_q=p1q,
            peak2_freq=p2f,
            peak2_gain=p2g,
            peak2_q=p2q,
            lowshelf_freq=lsf,
            lowshelf_gain=lsg,
            highshelf_freq=hsf,
            highshelf_gain=hsg,
        )

        x_proc = x.expand(-1, 2, -1) if in_ch == 1 else x[:, :2, :]
        x_group = F.pad(x_proc, (self.ir_length - 1, 0)).reshape(1, n_nodes * 2, -1)
        w_group = h.flip(-1).reshape(n_nodes * 2, 2, self.ir_length).to(dtype=x.dtype)
        y = conv1d(x_group, w_group, groups=n_nodes).reshape(n_nodes, 2, time_len)
        return y

    def parameter_size(self):
        n_delays = int(self.delays.numel())
        return {
            "b": (n_delays, 2),
            "c": (2, n_delays),
            "U": (n_delays, n_delays),
            "gamma": (self.num_decay_freq, 1 if self.delay_independent_decay else n_delays),
            "eq_peak1_freq": 1,
            "eq_peak1_gain": 1,
            "eq_peak1_Q": 1,
            "eq_peak2_freq": 1,
            "eq_peak2_gain": 1,
            "eq_peak2_Q": 1,
            "eq_lowshelf_freq": 1,
            "eq_lowshelf_gain": 1,
            "eq_highshelf_freq": 1,
            "eq_highshelf_gain": 1,
        }

    def decode_params(self, raw_params):
        n_delays = int(self.delays.numel())
        return {
            "b": raw_params["b"],
            "c": raw_params["c"],
            "U": self._orthogonal_from_raw(raw_params["U"].reshape(-1, n_delays, n_delays)),
            "gamma": self._activated_gamma(raw_params["gamma"], n_delays),
            "eq_peak1_freq": self._minmax(raw_params["eq_peak1_freq"], self.eq_peak1_min_freq, self.eq_peak1_max_freq),
            "eq_peak1_gain": raw_params["eq_peak1_gain"],
            "eq_peak1_Q": self._minmax(raw_params["eq_peak1_Q"], self.eq_peak1_min_q, self.eq_peak1_max_q),
            "eq_peak2_freq": self._minmax(raw_params["eq_peak2_freq"], self.eq_peak2_min_freq, self.eq_peak2_max_freq),
            "eq_peak2_gain": raw_params["eq_peak2_gain"],
            "eq_peak2_Q": self._minmax(raw_params["eq_peak2_Q"], self.eq_peak2_min_q, self.eq_peak2_max_q),
            "eq_lowshelf_freq": self._minmax(raw_params["eq_lowshelf_freq"], self.eq_lowshelf_min_freq, self.eq_lowshelf_max_freq),
            "eq_lowshelf_gain": raw_params["eq_lowshelf_gain"],
            "eq_highshelf_freq": self._minmax(raw_params["eq_highshelf_freq"], self.eq_highshelf_min_freq, self.eq_highshelf_max_freq),
            "eq_highshelf_gain": raw_params["eq_highshelf_gain"],
        }
