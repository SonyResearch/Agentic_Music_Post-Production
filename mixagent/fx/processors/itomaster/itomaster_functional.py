import dasp_pytorch
import torch
import torch.nn.functional as F
import torchcomp


EPS = 1e-6


def distortion(x: torch.Tensor, drive_db: torch.Tensor):
    """
    Adapted from ITO-Master's tanh soft-clipping distortion.

    This version is insert-only and leaves dry/wet mixing to the Grafx wrapper.
    """
    drive = drive_db.view(-1, 1, 1)
    return torch.tanh(x * torch.pow(10.0, drive / 20.0))


def _linkwitz_riley_biquad(
    cutoff_freq: torch.Tensor,
    sample_rate: int | float,
    filter_type: str,
):
    cutoff_freq = torch.as_tensor(cutoff_freq)
    q_factor = torch.ones_like(cutoff_freq) / torch.sqrt(
        torch.tensor(2.0, device=cutoff_freq.device, dtype=cutoff_freq.dtype)
    )
    gain_db = torch.zeros_like(cutoff_freq)
    return dasp_pytorch.signal.biquad(
        gain_db,
        cutoff_freq,
        q_factor,
        sample_rate,
        filter_type,
    )


def linkwitz_riley_4th_order(
    x: torch.Tensor,
    cutoff_freq: torch.Tensor,
    sample_rate: int | float,
    filter_type: str,
):
    b, a = _linkwitz_riley_biquad(cutoff_freq, sample_rate, filter_type)
    sos = torch.cat((b, a), dim=-1).unsqueeze(1)
    x_once = dasp_pytorch.signal.sosfilt_via_fsm(sos, x)
    return dasp_pytorch.signal.sosfilt_via_fsm(sos, x_once)


def _apply_compexp(
    band: torch.Tensor,
    sample_rate: int | float,
    comp_thresh: torch.Tensor,
    comp_ratio: torch.Tensor,
    exp_thresh: torch.Tensor,
    exp_ratio: torch.Tensor,
    comp_at: torch.Tensor,
    comp_rt: torch.Tensor,
    exp_at: torch.Tensor,
    exp_rt: torch.Tensor,
    expander_enabled: bool,
):
    try:
        sidechain = band.sum(dim=1).abs()
        if expander_enabled:
            gain = torchcomp.compexp_gain(
                sidechain,
                comp_thresh=comp_thresh,
                comp_ratio=comp_ratio,
                exp_thresh=exp_thresh,
                exp_ratio=exp_ratio,
                at=torchcomp.ms2coef(comp_at, sample_rate),
                rt=torchcomp.ms2coef(comp_rt, sample_rate),
            ).unsqueeze(1)
            return band * gain

        gain = torchcomp.compexp_gain(
            sidechain,
            comp_thresh=comp_thresh,
            comp_ratio=comp_ratio,
            exp_thresh=sidechain.new_full(comp_thresh.shape, -120.0),
            exp_ratio=torch.ones_like(exp_ratio),
            at=torchcomp.ms2coef(comp_at, sample_rate),
            rt=torchcomp.ms2coef(comp_rt, sample_rate),
        ).unsqueeze(1)
        return band * gain
    except Exception:
        return band


def _fast_control_rate_compexp_gain(
    sidechain: torch.Tensor,
    sample_rate: int | float,
    comp_thresh: torch.Tensor,
    comp_ratio: torch.Tensor,
    exp_thresh: torch.Tensor,
    exp_ratio: torch.Tensor,
    comp_at: torch.Tensor,
    comp_rt: torch.Tensor,
    exp_at: torch.Tensor,
    exp_rt: torch.Tensor,
    expander_enabled: bool,
    target_control_rate_hz: float = 3000.0,
):
    batch, num_samples = sidechain.shape
    hop = max(1, int(round(float(sample_rate) / float(target_control_rate_hz))))
    if hop > 1:
        control = F.max_pool1d(
            sidechain.unsqueeze(1),
            kernel_size=hop,
            stride=hop,
            ceil_mode=True,
        ).squeeze(1)
    else:
        control = sidechain

    zi = control.new_zeros(batch)
    env_cmp = torchcomp.compressor_core(control, zi, comp_rt, comp_at)
    env_db = 20.0 * torch.log10(env_cmp.clamp_min(EPS))

    comp_thresh = comp_thresh.unsqueeze(-1)
    comp_ratio = comp_ratio.unsqueeze(-1)
    exp_thresh = exp_thresh.unsqueeze(-1)
    exp_ratio = exp_ratio.unsqueeze(-1)

    zero = torch.zeros_like(env_db)
    comp_gain_db = torch.where(
        env_db > comp_thresh,
        -(1.0 - 1.0 / comp_ratio) * (env_db - comp_thresh),
        zero,
    )
    if expander_enabled:
        env_exp = torchcomp.compressor_core(control, zi, exp_rt, exp_at)
        env_exp_db = 20.0 * torch.log10(env_exp.clamp_min(EPS))
        exp_gain_db = torch.where(
            env_exp_db < exp_thresh,
            (1.0 / exp_ratio - 1.0) * (env_exp_db - exp_thresh),
            zero,
        )
    else:
        exp_gain_db = zero
    gain_control = torch.pow(10.0, (comp_gain_db + exp_gain_db) / 20.0)

    if hop > 1:
        gain = gain_control.repeat_interleave(hop, dim=-1)[..., :num_samples]
    else:
        gain = gain_control
    return gain


def _fast_apply_compexp(
    band: torch.Tensor,
    sample_rate: int | float,
    comp_thresh: torch.Tensor,
    comp_ratio: torch.Tensor,
    exp_thresh: torch.Tensor,
    exp_ratio: torch.Tensor,
    comp_at: torch.Tensor,
    comp_rt: torch.Tensor,
    exp_at: torch.Tensor,
    exp_rt: torch.Tensor,
    expander_enabled: bool,
    target_control_rate_hz: float = 3000.0,
):
    sidechain = band.sum(dim=1).abs()
    gain = _fast_control_rate_compexp_gain(
        sidechain=sidechain,
        sample_rate=sample_rate,
        comp_thresh=comp_thresh,
        comp_ratio=comp_ratio,
        exp_thresh=exp_thresh,
        exp_ratio=exp_ratio,
        comp_at=comp_at,
        comp_rt=comp_rt,
        exp_at=exp_at,
        exp_rt=exp_rt,
        expander_enabled=expander_enabled,
        target_control_rate_hz=target_control_rate_hz,
    )
    return band * gain.unsqueeze(1)


def multiband_compressor(
    x: torch.Tensor,
    sample_rate: int | float,
    low_cutoff: torch.Tensor,
    high_cutoff: torch.Tensor,
    low_shelf_comp_thresh: torch.Tensor,
    low_shelf_comp_ratio: torch.Tensor,
    low_shelf_exp_thresh: torch.Tensor,
    low_shelf_exp_ratio: torch.Tensor,
    low_shelf_comp_at: torch.Tensor,
    low_shelf_comp_rt: torch.Tensor,
    low_shelf_exp_at: torch.Tensor,
    low_shelf_exp_rt: torch.Tensor,
    mid_band_comp_thresh: torch.Tensor,
    mid_band_comp_ratio: torch.Tensor,
    mid_band_exp_thresh: torch.Tensor,
    mid_band_exp_ratio: torch.Tensor,
    mid_band_comp_at: torch.Tensor,
    mid_band_comp_rt: torch.Tensor,
    mid_band_exp_at: torch.Tensor,
    mid_band_exp_rt: torch.Tensor,
    high_shelf_comp_thresh: torch.Tensor,
    high_shelf_comp_ratio: torch.Tensor,
    high_shelf_exp_thresh: torch.Tensor,
    high_shelf_exp_ratio: torch.Tensor,
    high_shelf_comp_at: torch.Tensor,
    high_shelf_comp_rt: torch.Tensor,
    high_shelf_exp_at: torch.Tensor,
    high_shelf_exp_rt: torch.Tensor,
    expander_enabled: bool = False,
):
    bs, chs, seq_len = x.size()

    low_cutoff = low_cutoff.view(-1, 1, 1)
    high_cutoff = high_cutoff.view(-1, 1, 1)

    low_band = linkwitz_riley_4th_order(
        x, low_cutoff, sample_rate, filter_type="low_pass"
    )
    high_band = linkwitz_riley_4th_order(
        x, high_cutoff, sample_rate, filter_type="high_pass"
    )
    mid_band = x - low_band - high_band

    x_out_low = _apply_compexp(
        low_band,
        sample_rate,
        low_shelf_comp_thresh,
        low_shelf_comp_ratio,
        low_shelf_exp_thresh,
        low_shelf_exp_ratio,
        low_shelf_comp_at,
        low_shelf_comp_rt,
        low_shelf_exp_at,
        low_shelf_exp_rt,
        expander_enabled,
    )
    x_out_mid = _apply_compexp(
        mid_band,
        sample_rate,
        mid_band_comp_thresh,
        mid_band_comp_ratio,
        mid_band_exp_thresh,
        mid_band_exp_ratio,
        mid_band_comp_at,
        mid_band_comp_rt,
        mid_band_exp_at,
        mid_band_exp_rt,
        expander_enabled,
    )
    x_out_high = _apply_compexp(
        high_band,
        sample_rate,
        high_shelf_comp_thresh,
        high_shelf_comp_ratio,
        high_shelf_exp_thresh,
        high_shelf_exp_ratio,
        high_shelf_comp_at,
        high_shelf_comp_rt,
        high_shelf_exp_at,
        high_shelf_exp_rt,
        expander_enabled,
    )

    x_out = x_out_low + x_out_mid + x_out_high
    return x_out.view(bs, chs, seq_len)


def fast_multiband_compressor(
    x: torch.Tensor,
    sample_rate: int | float,
    low_cutoff: torch.Tensor,
    high_cutoff: torch.Tensor,
    low_shelf_comp_thresh: torch.Tensor,
    low_shelf_comp_ratio: torch.Tensor,
    low_shelf_exp_thresh: torch.Tensor,
    low_shelf_exp_ratio: torch.Tensor,
    low_shelf_comp_at: torch.Tensor,
    low_shelf_comp_rt: torch.Tensor,
    low_shelf_exp_at: torch.Tensor,
    low_shelf_exp_rt: torch.Tensor,
    mid_band_comp_thresh: torch.Tensor,
    mid_band_comp_ratio: torch.Tensor,
    mid_band_exp_thresh: torch.Tensor,
    mid_band_exp_ratio: torch.Tensor,
    mid_band_comp_at: torch.Tensor,
    mid_band_comp_rt: torch.Tensor,
    mid_band_exp_at: torch.Tensor,
    mid_band_exp_rt: torch.Tensor,
    high_shelf_comp_thresh: torch.Tensor,
    high_shelf_comp_ratio: torch.Tensor,
    high_shelf_exp_thresh: torch.Tensor,
    high_shelf_exp_ratio: torch.Tensor,
    high_shelf_comp_at: torch.Tensor,
    high_shelf_comp_rt: torch.Tensor,
    high_shelf_exp_at: torch.Tensor,
    high_shelf_exp_rt: torch.Tensor,
    expander_enabled: bool = False,
    target_control_rate_hz: float = 3000.0,
):
    bs, chs, seq_len = x.size()

    low_cutoff = low_cutoff.view(-1, 1, 1)
    high_cutoff = high_cutoff.view(-1, 1, 1)

    low_band = linkwitz_riley_4th_order(
        x, low_cutoff, sample_rate, filter_type="low_pass"
    )
    high_band = linkwitz_riley_4th_order(
        x, high_cutoff, sample_rate, filter_type="high_pass"
    )
    mid_band = x - low_band - high_band

    x_out_low = _fast_apply_compexp(
        low_band,
        sample_rate,
        low_shelf_comp_thresh,
        low_shelf_comp_ratio,
        low_shelf_exp_thresh,
        low_shelf_exp_ratio,
        low_shelf_comp_at,
        low_shelf_comp_rt,
        low_shelf_exp_at,
        low_shelf_exp_rt,
        expander_enabled,
        target_control_rate_hz=target_control_rate_hz,
    )
    x_out_mid = _fast_apply_compexp(
        mid_band,
        sample_rate,
        mid_band_comp_thresh,
        mid_band_comp_ratio,
        mid_band_exp_thresh,
        mid_band_exp_ratio,
        mid_band_comp_at,
        mid_band_comp_rt,
        mid_band_exp_at,
        mid_band_exp_rt,
        expander_enabled,
        target_control_rate_hz=target_control_rate_hz,
    )
    x_out_high = _fast_apply_compexp(
        high_band,
        sample_rate,
        high_shelf_comp_thresh,
        high_shelf_comp_ratio,
        high_shelf_exp_thresh,
        high_shelf_exp_ratio,
        high_shelf_comp_at,
        high_shelf_comp_rt,
        high_shelf_exp_at,
        high_shelf_exp_rt,
        expander_enabled,
        target_control_rate_hz=target_control_rate_hz,
    )

    x_out = x_out_low + x_out_mid + x_out_high
    return x_out.view(bs, chs, seq_len)


def limiter(
    x: torch.Tensor,
    sample_rate: int | float,
    threshold: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
):
    """
    Adapted from ITO-Master's limiter.

    The Grafx adapter now decodes attack and release to bounded smoothing
    coefficients before calling this function, matching the current Diffvox-style
    runtime parameterization.
    """
    gain = torchcomp.limiter_gain(
        x.sum(dim=1).abs(),
        threshold=threshold,
        at=at,
        rt=rt,
    ).unsqueeze(1)
    return x * gain


def fast_limiter(
    x: torch.Tensor,
    sample_rate: int | float,
    threshold: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
    target_control_rate_hz: float = 1000.0,
):
    """
    Faster Grafx-oriented limiter variant.

    Differences from the exact ITO-Master-style path:
    - uses a max sidechain across channels instead of summed magnitude
    - computes the detector and gain at control rate, then upsamples to audio rate
    - uses a single smoothing pass on the control-rate envelope
    """
    batch, _, num_samples = x.shape

    sidechain = x.abs().amax(dim=1)

    hop = max(1, int(round(float(sample_rate) / float(target_control_rate_hz))))
    if hop > 1:
        control = F.max_pool1d(
            sidechain.unsqueeze(1),
            kernel_size=hop,
            stride=hop,
            ceil_mode=True,
        ).squeeze(1)
    else:
        control = sidechain

    control_sr = float(sample_rate) / float(hop)
    # at/rt already arrive as bounded smoothing coefficients from the adapter.
    at_coef = at
    rt_coef = rt
    zi = control.new_zeros(batch)
    env = torchcomp.compressor_core(control, zi, rt_coef, at_coef)

    threshold_amp = torchcomp.db2amp(threshold).unsqueeze(-1)
    gain_control = torch.minimum(
        torch.ones_like(env),
        threshold_amp / env.clamp_min(EPS),
    )
    gain_control = F.relu(gain_control)

    if hop > 1:
        gain = gain_control.repeat_interleave(hop, dim=-1)[..., :num_samples]
    else:
        gain = gain_control

    return x * gain.unsqueeze(1)
