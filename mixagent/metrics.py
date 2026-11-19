"""Intrusive mix metrics against the ground-truth mix.

* MRS -- multi-resolution STFT distance (``auraloss.freq.MultiResolutionSTFTLoss``,
  spectral convergence + log magnitude, FFT sizes 1024/2048/8192), averaged over
  {left, right} and {mid, side} (mid = (L+R)/2, side = (L-R)/2).
* MSS -- DDSP-style mid/side multi-scale spectral loss (FFT 4096/2048/1024/512;
  0.9 * L1 magnitude + 0.1 * MSE log10 magnitude; mid = L+R, side = L-R, weight 0.5).

Before scoring, the prediction is loudness-normalized to -12 LUFS (pyloudnorm, no
peak guard); the GT mixes of the datasets are already at -12 LUFS.
"""

from __future__ import annotations

import math

import numpy as np

MRS_RESOLUTIONS = ((1024, 256, 1024), (2048, 512, 2048), (8192, 2048, 8192))
MSS_RESOLUTIONS = ((4096, 1024, 4096), (2048, 512, 2048), (1024, 256, 1024), (512, 128, 512))
MSS_MID_WEIGHT = 0.5
MSS_LOG_MAG_WEIGHT = 0.1
MSS_EPS = 1e-7
FRONTEND_MAGNITUDE_EPS = 1e-7
EPS = 1e-8
SCORE_LUFS = -12.0


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.ascontiguousarray(audio.astype(np.float32, copy=False))


def loudness_normalize(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """Gain to ``target_lufs`` integrated LUFS (BS.1770), no peak guard; silent or
    unmeasurable audio is returned unchanged."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or float(np.max(np.abs(arr))) <= 1e-6:
        return arr
    import pyloudnorm as pyln

    loudness = float(pyln.Meter(int(sample_rate)).integrated_loudness(arr + 1e-10))
    if not math.isfinite(loudness):
        return arr
    return (arr * (10.0 ** ((float(target_lufs) - loudness) / 20.0))).astype(np.float32,
                                                                            copy=False)


def _to_tensor(audio: np.ndarray, device: str = "cpu"):
    import torch

    return torch.from_numpy(np.ascontiguousarray(audio.T)).float().unsqueeze(0).to(device)


def _stft_loss_total(pred, target, sample_rate: int) -> float:
    import auraloss
    import torch

    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[r[0] for r in MRS_RESOLUTIONS], hop_sizes=[r[1] for r in MRS_RESOLUTIONS],
        win_lengths=[r[2] for r in MRS_RESOLUTIONS], window="hann_window",
        w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0, w_phs=0.0, sample_rate=sample_rate,
        output="full", reduction="mean", eps=EPS).to(pred.device)
    loss_fn.eval()
    with torch.inference_mode():
        total = loss_fn(pred, target)[0]
    return float(total.detach().cpu())


def mrs(pred, target, sample_rate: int) -> float:
    """MRS of two [1, 2, T] tensors."""
    lp, rp = pred[:, 0:1, :], pred[:, 1:2, :]
    lt, rt = target[:, 0:1, :], target[:, 1:2, :]
    lr = np.mean([_stft_loss_total(lp, lt, sample_rate), _stft_loss_total(rp, rt, sample_rate)])
    ms = np.mean([_stft_loss_total(0.5 * (lp + rp), 0.5 * (lt + rt), sample_rate),
                  _stft_loss_total(0.5 * (lp - rp), 0.5 * (lt - rt), sample_rate)])
    return float(np.mean([float(lr), float(ms)]))


def _frontend_magnitude(signal, fft_size, hop_size, win_length):
    import torch

    window = torch.hann_window(win_length, device=signal.device, dtype=signal.dtype)
    spec = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
                      window=window, return_complex=True)
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + FRONTEND_MAGNITUDE_EPS)
    if signal.shape[-1] % round(fft_size / 4) == 0:
        mag = mag[:, :, :-1]
    if fft_size % 2 == 0:
        mag = mag[:, 1:]
    return mag


def mss(pred, target) -> float:
    """MSS of two [1, 2, T] tensors."""
    import torch
    import torch.nn.functional as F

    pm, ps = pred[:, 0, :] + pred[:, 1, :], pred[:, 0, :] - pred[:, 1, :]
    tm, ts = target[:, 0, :] + target[:, 1, :], target[:, 0, :] - target[:, 1, :]
    total_mag = torch.zeros((), device=pred.device, dtype=pred.dtype)
    total_log = torch.zeros((), device=pred.device, dtype=pred.dtype)
    with torch.inference_mode():
        for fft, hop, win in MSS_RESOLUTIONS:
            pmm, psm = _frontend_magnitude(pm, fft, hop, win), _frontend_magnitude(ps, fft, hop, win)
            tmm, tsm = _frontend_magnitude(tm, fft, hop, win), _frontend_magnitude(ts, fft, hop, win)
            mag = (MSS_MID_WEIGHT * F.l1_loss(pmm, tmm)
                   + (1.0 - MSS_MID_WEIGHT) * F.l1_loss(psm, tsm))
            log = (MSS_MID_WEIGHT * F.mse_loss(torch.log10(pmm + MSS_EPS), torch.log10(tmm + MSS_EPS))
                   + (1.0 - MSS_MID_WEIGHT)
                   * F.mse_loss(torch.log10(psm + MSS_EPS), torch.log10(tsm + MSS_EPS)))
            total_mag = total_mag + mag
            total_log = total_log + log
        total = (1.0 - MSS_LOG_MAG_WEIGHT) * total_mag + MSS_LOG_MAG_WEIGHT * total_log
    return float(total.detach().cpu())


def mrs_arrays(pred: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    """MRS of two aligned [T, C] arrays (trimmed to the common length)."""
    pred, target = ensure_stereo(pred), ensure_stereo(target)
    n = min(pred.shape[0], target.shape[0])
    return mrs(_to_tensor(pred[:n]), _to_tensor(target[:n]), sample_rate)


# --------------------------------------------------------------------------- AF
# Audio-feature distance (weights: rms 0.1, crest 0.001, stereo width 1.0,
# stereo imbalance 1.0, 24-band Bark spectrum of mid/side 0.1; MSE per feature).
AF_WEIGHTS = (0.1, 0.001, 1.0, 1.0, 0.1)


def _rms(x):
    import torch

    return torch.sqrt(torch.mean(x ** 2, dim=-1).clamp(min=1e-8))


def _crest_factor(x):
    import torch

    peak = torch.max(torch.abs(x), dim=-1)[0]
    return 20.0 * torch.log10((peak / _rms(x).clamp(min=1e-8)).clamp(min=1e-8))


def _stereo_width(x):
    import torch

    mid, side = x[:, 0, :] + x[:, 1, :], x[:, 0, :] - x[:, 1, :]
    return torch.mean(side ** 2, dim=-1) / torch.mean(mid ** 2, dim=-1).clamp(min=1e-8)


def _stereo_imbalance(x):
    import torch

    left, right = torch.mean(x[:, 0, :] ** 2, dim=-1), torch.mean(x[:, 1, :] ** 2, dim=-1)
    return (right - left) / (right + left).clamp(min=1e-8)


def _hz_to_bark(freq: float) -> float:
    bark = ((26.81 * freq) / (1960.0 + freq)) - 0.53
    if bark < 2:
        bark += 0.15 * (2 - bark)
    elif bark > 20.1:
        bark += 0.22 * (bark - 20.1)
    return float(bark)


def _bark_to_hz(barks):
    barks = barks.clone()
    if bool((barks < 2).any()):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif bool((barks > 20.1).any()):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22
    return 1960.0 * ((barks + 0.53) / (26.28 - barks))


def _bark_filterbank(fft_size, n_bands, sample_rate, f_min, f_max, device, dtype):
    import torch

    freqs = torch.linspace(0.0, sample_rate // 2, fft_size // 2 + 1, device=device, dtype=dtype)
    bark_points = torch.linspace(_hz_to_bark(float(f_min)), _hz_to_bark(float(f_max)),
                                 n_bands + 2, device=device, dtype=dtype)
    freq_points = _bark_to_hz(bark_points)
    freq_diffs = freq_points[1:] - freq_points[:-1]
    slopes = freq_points.unsqueeze(0) - freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / freq_diffs[:-1]
    up = slopes[:, 2:] / freq_diffs[1:]
    return torch.maximum(torch.zeros(1, device=device, dtype=dtype), torch.minimum(down, up))


def _barkspectrum(x, sample_rate, fft_size=32768, n_bands=24, f_min=20.0, f_max=20000.0):
    import torch

    f_max = min(float(f_max), float(sample_rate) / 2.0)
    fb = _bark_filterbank(fft_size, n_bands, sample_rate, f_min, f_max, x.device, x.dtype)
    outputs = []
    for signal in (x[:, 0, :] + x[:, 1, :], x[:, 0, :] - x[:, 1, :]):
        window = torch.hann_window(fft_size, device=signal.device, dtype=signal.dtype)
        mag = torch.abs(torch.stft(signal, n_fft=fft_size, hop_length=fft_size // 4,
                                   win_length=fft_size, window=window, return_complex=True))
        outputs.append(torch.log(torch.matmul(torch.mean(mag, dim=-1), fb) + 1e-8))
    return torch.stack(outputs, dim=-1)


def af(pred, target, sample_rate: int) -> float:
    """AF distance of two [1, 2, T] tensors."""
    import torch
    import torch.nn.functional as F

    transforms = (_rms, _crest_factor, _stereo_width, _stereo_imbalance,
                  lambda x: _barkspectrum(x, sample_rate))
    total = 0.0
    with torch.inference_mode():
        for transform, weight in zip(transforms, AF_WEIGHTS):
            total += float((weight * F.mse_loss(transform(pred), transform(target))).detach().cpu())
    return float(total)
