import torch
import torch.nn.functional as F
from torchcomp import compexp_gain, db2amp


from torchlpc import sample_wise_lpc
from typing import List, Tuple, Union, Any, Optional
import math


def inv_22(a, b, c, d):
    return torch.stack([d, -b, -c, a]).view(2, 2) / (a * d - b * c)


def eig_22(a, b, c, d):
    # https://croninprojects.org/Vince/Geodesy/FindingEigenvectors.pdf
    T = a + d
    D = a * d - b * c
    half_T = T * 0.5
    root = torch.sqrt(half_T * half_T - D)  # + 0j)
    L = torch.stack([half_T + root, half_T - root])

    y = (L - a) / b
    # y = c / L
    V = torch.stack([torch.ones_like(y), y])
    return L, V / V.abs().square().sum(0).sqrt()


def fir(x, b):
    padded = F.pad(x.reshape(-1, 1, x.size(-1)), (b.size(0) - 1, 0))
    return F.conv1d(padded, b.flip(0).view(1, 1, -1)).view(*x.shape)


def allpole(x: torch.Tensor, a: torch.Tensor):
    h = x.reshape(-1, x.shape[-1])
    return sample_wise_lpc(
        h,
        a.broadcast_to(h.shape + a.shape),
    ).reshape(*x.shape)


def biquad(x: torch.Tensor, b0, b1, b2, a0, a1, a2):
    b0 = b0 / a0
    b1 = b1 / a0
    b2 = b2 / a0
    a1 = a1 / a0
    a2 = a2 / a0

    beta1 = b1 - b0 * a1
    beta2 = b2 - b0 * a2

    tmp = a1.square() - 4 * a2

    def _complex_branch(x_loc, beta1_loc, beta2_loc, a1_loc, tmp_loc):
        pole = 0.5 * (-a1_loc + 1j * torch.sqrt(-tmp_loc))
        u = -1j * x_loc[..., :-1]
        pole_for_u = pole
        while pole_for_u.ndim < u.ndim:
            pole_for_u = pole_for_u.unsqueeze(-1)
        h_loc = sample_wise_lpc(
            u.reshape(-1, u.shape[-1]),
            -pole_for_u.broadcast_to(u.shape).reshape(-1, u.shape[-1], 1),
        ).reshape(*u.shape)
        gain_term = beta1_loc * pole.real / pole.imag + beta2_loc / pole.imag
        beta1_for_h = beta1_loc
        gain_for_h = gain_term
        while beta1_for_h.ndim < h_loc.ndim:
            beta1_for_h = beta1_for_h.unsqueeze(-1)
        while gain_for_h.ndim < h_loc.ndim:
            gain_for_h = gain_for_h.unsqueeze(-1)
        h_loc = (
            h_loc.real * gain_for_h
            - beta1_for_h * h_loc.imag
        )
        return h_loc

    def _real_branch(x_loc, beta1_loc, beta2_loc, a1_loc, a2_loc):
        L, V = eig_22(-a1_loc, -a2_loc, torch.ones_like(a1_loc), torch.zeros_like(a1_loc))
        inv_V = inv_22(*V.view(-1))
        C = torch.stack([beta1_loc, beta2_loc]) @ V

        h_loc = x_loc[..., :-1].unsqueeze(-2) * inv_V[:, :1]
        L = L.unsqueeze(-1).broadcast_to(h_loc.shape)
        h_loc = (
            sample_wise_lpc(h_loc.reshape(-1, h_loc.shape[-1]), -L.reshape(-1, L.shape[-1], 1))
            .reshape(*h_loc.shape)
            .transpose(-2, -1)
        ) @ C
        return h_loc

    if tmp.ndim == 0:
        if bool(tmp < 0):
            h = _complex_branch(x, beta1, beta2, a1, tmp)
        else:
            h = _real_branch(x, beta1, beta2, a1, a2)
    else:
        if x.ndim != 2:
            raise RuntimeError(
                f"Vectorized biquad currently expects x shape [B, T], got {tuple(x.shape)}"
            )
        neg_mask = tmp < 0
        if bool(torch.all(neg_mask)):
            # Fast batched path for the common underdamped case.
            h = _complex_branch(x, beta1, beta2, a1, tmp)
        else:
            # Mixed/real-pole fallback keeps behavior stable.
            h = torch.empty_like(x[..., :-1])
            if bool(torch.any(neg_mask)):
                h[neg_mask] = _complex_branch(
                    x[neg_mask], beta1[neg_mask], beta2[neg_mask], a1[neg_mask], tmp[neg_mask]
                )
            pos_mask = ~neg_mask
            if bool(torch.any(pos_mask)):
                h_rows = []
                x_pos = x[pos_mask]
                beta1_pos = beta1[pos_mask]
                beta2_pos = beta2[pos_mask]
                a1_pos = a1[pos_mask]
                a2_pos = a2[pos_mask]
                for i in range(x_pos.shape[0]):
                    hi = _real_branch(
                        x_pos[i : i + 1],
                        beta1_pos[i],
                        beta2_pos[i],
                        a1_pos[i],
                        a2_pos[i],
                    )
                    h_rows.append(hi)
                h[pos_mask] = torch.cat(h_rows, dim=0)

    b0_for_x = b0
    while b0_for_x.ndim < x.ndim:
        b0_for_x = b0_for_x.unsqueeze(-1)
    tmp = b0_for_x * x
    y = torch.cat([tmp[..., :1], h + tmp[..., 1:]], -1)
    return y


def highpass_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    Q: torch.Tensor,
):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2.0 / Q

    b0 = (1 + torch.cos(w0)) / 2
    b1 = -1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return b0, b1, b2, a0, a1, a2


def apply_biquad(bq):
    return lambda waveform, *args, **kwargs: biquad(waveform, *bq(*args, **kwargs))


highpass_biquad = apply_biquad(highpass_biquad_coef)


def lowpass_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    Q: torch.Tensor,
):
    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q

    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return b0, b1, b2, a0, a1, a2


def equalizer_biquad_coef(
    sample_rate: int,
    center_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * center_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q

    b0 = 1 + alpha * A
    b1 = -2 * torch.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def lowshelf_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    cosw0 = torch.cos(w0)
    sqrtA = torch.sqrt(A)

    b0 = A * (A + 1 - (A - 1) * cosw0 + 2 * alpha * sqrtA)
    b1 = 2 * A * (A - 1 - (A + 1) * cosw0)
    b2 = A * (A + 1 - (A - 1) * cosw0 - 2 * alpha * sqrtA)

    a0 = A + 1 + (A - 1) * cosw0 + 2 * alpha * sqrtA
    a1 = -2 * (A - 1 + (A + 1) * cosw0)
    a2 = A + 1 + (A - 1) * cosw0 - 2 * alpha * sqrtA

    return b0, b1, b2, a0, a1, a2


def highshelf_biquad_coef(
    sample_rate: int,
    cutoff_freq: torch.Tensor,
    gain: torch.Tensor,
    Q: torch.Tensor,
):

    w0 = 2 * torch.pi * cutoff_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    cosw0 = torch.cos(w0)
    sqrtA = torch.sqrt(A)

    b0 = A * (A + 1 + (A - 1) * cosw0 + 2 * alpha * sqrtA)
    b1 = -2 * A * (A - 1 + (A + 1) * cosw0)
    b2 = A * (A + 1 + (A - 1) * cosw0 - 2 * alpha * sqrtA)

    a0 = A + 1 - (A - 1) * cosw0 + 2 * alpha * sqrtA
    a1 = 2 * (A - 1 - (A + 1) * cosw0)
    a2 = A + 1 - (A - 1) * cosw0 - 2 * alpha * sqrtA

    return b0, b1, b2, a0, a1, a2


highpass_biquad = apply_biquad(highpass_biquad_coef)
lowpass_biquad = apply_biquad(lowpass_biquad_coef)
highshelf_biquad = apply_biquad(highshelf_biquad_coef)
lowshelf_biquad = apply_biquad(lowshelf_biquad_coef)
equalizer_biquad = apply_biquad(equalizer_biquad_coef)


def avg(rms: torch.Tensor, avg_coef: torch.Tensor):
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)
    # Match original diffvox behavior while supporting avg_coef passed as [B].
    if avg_coef.dim() == 1:
        avg_coef = avg_coef.unsqueeze(-1)

    h = rms * avg_coef

    return sample_wise_lpc(
        h,
        (avg_coef - 1).broadcast_to(h.shape).unsqueeze(-1),
    )
#torch 2.9.1+cu128
# torchlpc 0.7.2

def avg_rms(audio: torch.Tensor, avg_coef) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def compressor_expander(
    x: torch.Tensor,
    avg_coef: Union[torch.Tensor, float],
    cmp_th: Union[torch.Tensor, float],
    cmp_ratio: Union[torch.Tensor, float],
    exp_th: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    at: Union[torch.Tensor, float],
    rt: Union[torch.Tensor, float],
    make_up: torch.Tensor,
    lookahead_func=lambda x: x,
):
    rms = avg_rms(x, avg_coef=avg_coef)
    gain = compexp_gain(rms, cmp_th, cmp_ratio, exp_th, exp_ratio, at, rt)
    gain = lookahead_func(gain)
    make_up_amp = db2amp(make_up)
    if make_up_amp.dim() == 1:
        make_up_amp = make_up_amp.unsqueeze(-1)
    return x * gain * make_up_amp.broadcast_to(x.shape[0], 1)
