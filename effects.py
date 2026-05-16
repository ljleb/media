import math
import torch
from typing import Optional


def dilate2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int] = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """
    Morphologically dilate values over the last two axes of a tensor.

    Args:
        x: Tensor of shape (..., H, W).
        kernel_size: Dilation window size. Usually 3, 5, etc.
        iterations: Number of repeated dilations.

    Returns:
        Tensor with same shape as x.
    """
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions: (..., H, W)")

    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel_size should be odd to preserve shape exactly")

    orig_shape = x.shape
    h, w = orig_shape[-2:]

    # Collapse all leading dims into N, add a channel dim: (N, 1, H, W)
    y = x.reshape(-1, 1, h, w)

    padding = (kh // 2, kw // 2)

    for _ in range(iterations):
        y = torch.nn.functional.max_pool2d(
            y,
            kernel_size=(kh, kw),
            stride=1,
            padding=padding,
        )

    return y.reshape(orig_shape)


def color(v=0.0, **kwargs):
    r = g = b = v
    if "r" in kwargs:
        r = kwargs.pop("r")
    if "g" in kwargs:
        g = kwargs.pop("g")
    if "b" in kwargs:
        b = kwargs.pop("b")
    if kwargs:
        raise TypeError(f"invalid arguments to vec(): {list(kwargs.keys())}")
    return torch.tensor((r, g, b), device=device, dtype=dtype).view(3, 1, 1)


Number = int | float


def _infer_dtype_device(
    *values: torch.Tensor | Number,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> tuple[torch.dtype, torch.device]:
    tensor_values = [x for x in values if isinstance(x, torch.Tensor)]

    if dtype is None:
        tensor_dtypes = {x.dtype for x in tensor_values}

        if len(tensor_dtypes) > 1:
            raise ValueError(
                f"Input tensors must share dtype unless dtype is specified; got {tensor_dtypes}"
            )

        dtype = tensor_values[0].dtype if tensor_values else torch.float32

        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.float32

    if device is None:
        tensor_devices = {x.device for x in tensor_values}

        if len(tensor_devices) > 1:
            raise ValueError(
                f"Input tensors must share device unless device is specified; got {tensor_devices}"
            )

        device = tensor_values[0].device if tensor_values else torch.device("cpu")
    else:
        device = torch.device(device)

    return dtype, device


def make_hsv(
    h: torch.Tensor | Number,
    s: torch.Tensor | Number,
    v: torch.Tensor | Number,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """
    Create a 3-channel HSV tensor from broadcastable h, s, v values.

    Args:
        h: Hue in [0, 1], tensor or scalar.
        s: Saturation in [0, 1], tensor or scalar.
        v: Value/brightness in [0, 1], tensor or scalar.
        dtype: Optional output dtype.
        device: Optional output device.

    Returns:
        Tensor shaped [3, ...], where:
            hsv[0] = hue
            hsv[1] = saturation
            hsv[2] = value
    """
    dtype, device = _infer_dtype_device(h, s, v, dtype=dtype, device=device)

    h = torch.as_tensor(h, dtype=dtype, device=device)
    s = torch.as_tensor(s, dtype=dtype, device=device)
    v = torch.as_tensor(v, dtype=dtype, device=device)

    h, s, v = torch.broadcast_tensors(h, s, v)

    return torch.stack([h, s, v], dim=0)


def oklch_to_rgb(
    l: torch.Tensor | float,
    c: torch.Tensor | float,
    h: torch.Tensor | float,
) -> torch.Tensor:
    """
    Convert OKLCH to sRGB.

    Args:
        l: lightness, usually around [0, 1]
        c: chroma, usually around [0, 0.3] for safe sRGB use
        h: hue, normalized so 1.0 = one full turn

    Returns:
        Tensor shaped [3, ...] in sRGB space, clamped to [0, 1].
    """
    dtype, device = _infer_dtype_device(l, c, h)
    if not isinstance(l, torch.Tensor):
        l = torch.as_tensor(l, device=device, dtype=dtype)
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c, device=device, dtype=dtype)
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h, device=device, dtype=dtype)

    l, c, h = torch.broadcast_tensors(l, c, h)

    angle = h * (2 * torch.pi)
    a = c * torch.cos(angle)
    b = c * torch.sin(angle)

    # OKLab -> LMS'
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b

    # LMS' -> LMS
    l3 = l_ ** 3
    m3 = m_ ** 3
    s3 = s_ ** 3

    # linear RGB
    r_lin = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    rgb_lin = torch.stack([r_lin, g_lin, b_lin], dim=0)

    # linear RGB -> sRGB
    threshold = 0.0031308
    rgb = torch.where(
        rgb_lin <= threshold,
        12.92 * rgb_lin,
        1.055 * torch.clamp_min(rgb_lin, 0.0).pow(1 / 2.4) - 0.055,
    )

    return rgb.clamp(0.0, 1.0)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3-channel HSV tensor to RGB.

    Args:
        hsv: Tensor shaped [3, ...], with:
            hsv[0] = hue in [0, 1]
            hsv[1] = saturation in [0, 1]
            hsv[2] = value in [0, 1]

    Returns:
        RGB tensor shaped [3, ...].
    """
    if not isinstance(hsv, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(hsv)!r}")

    if hsv.ndim < 1:
        raise ValueError(f"Expected HSV tensor with shape [3, ...], got scalar")

    if hsv.shape[0] != 3:
        raise ValueError(f"Expected HSV tensor with 3 channels in dim 0, got shape {tuple(hsv.shape)}")

    if not torch.is_floating_point(hsv):
        hsv = hsv.float()

    h, s, v = hsv.unbind(dim=0)

    h = h % 1.0
    s = s.clamp(0.0, 1.0)
    v = v.clamp(0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = torch.empty_like(hsv)
    r, g, b = rgb.unbind(dim=0)

    mask = i == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]

    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]

    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]

    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]

    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]

    mask = i == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return rgb


def hsv_to_r(hsv: torch.Tensor) -> torch.Tensor:
    if hsv.ndim < 3 or hsv.shape[-3] != 3:
        raise ValueError("hsv must have shape [..., 3, H, W]")
    if not torch.is_floating_point(hsv):
        hsv = hsv.float()

    h, s, v = hsv.unbind(dim=-3)
    h = h % 1.0
    s = s.clamp(0.0, 1.0)
    v = v.clamp(0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.empty_like(v)
    mask = i == 0
    r[mask] = v[mask]
    mask = i == 1
    r[mask] = q[mask]
    mask = i == 2
    r[mask] = p[mask]
    mask = i == 3
    r[mask] = p[mask]
    mask = i == 4
    r[mask] = t[mask]
    mask = i == 5
    r[mask] = v[mask]

    return r.unsqueeze(-3)


def hsv_to_g(hsv: torch.Tensor) -> torch.Tensor:
    if hsv.ndim < 3 or hsv.shape[-3] != 3:
        raise ValueError("hsv must have shape [..., 3, H, W]")
    if not torch.is_floating_point(hsv):
        hsv = hsv.float()

    h, s, v = hsv.unbind(dim=-3)
    h = h % 1.0
    s = s.clamp(0.0, 1.0)
    v = v.clamp(0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    g = torch.empty_like(v)
    mask = i == 0
    g[mask] = t[mask]
    mask = i == 1
    g[mask] = v[mask]
    mask = i == 2
    g[mask] = v[mask]
    mask = i == 3
    g[mask] = q[mask]
    mask = i == 4
    g[mask] = p[mask]
    mask = i == 5
    g[mask] = p[mask]

    return g.unsqueeze(-3)


def hsv_to_b(hsv: torch.Tensor) -> torch.Tensor:
    if hsv.ndim < 3 or hsv.shape[-3] != 3:
        raise ValueError("hsv must have shape [..., 3, H, W]")
    if not torch.is_floating_point(hsv):
        hsv = hsv.float()

    h, s, v = hsv.unbind(dim=-3)
    h = h % 1.0
    s = s.clamp(0.0, 1.0)
    v = v.clamp(0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    b = torch.empty_like(v)
    mask = i == 0
    b[mask] = p[mask]
    mask = i == 1
    b[mask] = p[mask]
    mask = i == 2
    b[mask] = t[mask]
    mask = i == 3
    b[mask] = v[mask]
    mask = i == 4
    b[mask] = v[mask]
    mask = i == 5
    b[mask] = q[mask]

    return b.unsqueeze(-3)


def hsv_to(idx: int, hsv: torch.Tensor) -> torch.Tensor:
    return {
        0: hsv_to_r,
        1: hsv_to_g,
        2: hsv_to_b,
    }[idx](hsv)


def tv_static_mask(
    image: torch.Tensor,
    amount: float = 0.3,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if image.ndim < 3:
        raise ValueError("image must have shape [..., C, H, W]")

    amount = float(max(0.0, min(1.0, amount)))
    leading = image.shape[:-3]
    h, w = image.shape[-2:]
    mask = torch.rand(*leading, 1, h, w, device=image.device, dtype=image.dtype, generator=generator)
    return (mask < amount).to(image.dtype)


def wave(
    uv: torch.Tensor,
    pattern: float = 0.5,
    skew: float = 0.5,
) -> torch.Tensor:
    if uv.ndim < 3 or uv.shape[-3] != 2:
        raise ValueError("uv must have shape [..., 2, H, W]")

    h, w = uv.shape[-2:]
    u, v = uv.unbind(dim=-3)

    u_px = u * max(w - 1, 1)
    wave_frequency = 0.01 + float(pattern) * 0.03
    wave_amplitude_px = h * (0.02 + float(pattern) * 0.08)
    distortion_v = torch.sin(u_px * wave_frequency) * (wave_amplitude_px / max(h - 1, 1))

    skew_angle = float(skew) * 30.0 * math.pi / 180.0
    x_centered_px = u_px - (w - 1) * 0.5
    shear_v = math.tan(skew_angle) * x_centered_px / max(h - 1, 1)

    return torch.stack((u, v + distortion_v + shear_v), dim=-3)


def scanlines(
    image: torch.Tensor,
    spacing: int = 10,
    thickness: int = 2,
    radial: bool = False,
) -> torch.Tensor:
    """
    Approximate the original canvas scanlines effect:
      1. draw black scanline geometry
      2. apply layer opacity
      3. composite over image with overlay blend mode

    Args:
        image: Tensor of shape [..., 3, H, W], usually in [0, 1].
        spacing: Distance between scanlines in pixels.
        thickness: Scanline thickness in pixels.
        opacity: Scanline layer opacity in [0, 1].
        radial: If False, horizontal scanlines. If True, radial/spoked scanlines.

    Returns:
        Tensor of shape [..., 3, H, W].
    """
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("image must have shape [..., 3, H, W]")
    if not torch.is_floating_point(image):
        image = image.float()

    spacing = max(1, int(spacing))
    thickness = max(1, int(thickness))

    h, w = image.shape[-2:]
    device = image.device
    dtype = image.dtype
    leading_ndim = image.ndim - 3
    pad = [1] * leading_ndim

    if not radial:
        y = torch.arange(h, device=device).view(*pad, 1, h, 1)
        mask = (y.remainder(spacing) < thickness).to(dtype)
    else:
        yy = torch.arange(h, device=device, dtype=dtype).view(*pad, 1, h, 1)
        xx = torch.arange(w, device=device, dtype=dtype).view(*pad, 1, 1, w)

        cy = (h - 1) * 0.5
        cx = (w - 1) * 0.5

        dy = yy - cy
        dx = xx - cx

        radius = torch.sqrt(dx * dx + dy * dy)
        theta = torch.atan2(dy, dx)

        # Original radial version loops every 2 degrees.
        sector = math.radians(2.0)
        angular_distance = torch.remainder(theta + 0.5 * sector, sector) - 0.5 * sector

        # Approximate perpendicular distance from nearest spoke.
        spoke_distance = radius * angular_distance.abs()

        # Original draws repeated short filled rectangles along each rotated spoke.
        dash_mask = radius.remainder(spacing) < max(1.0, spacing - 1.0)
        mask = ((spoke_distance < (0.5 * thickness)) & dash_mask).to(dtype)

    # Canvas overlay blend with a black source layer:
    #   overlay(base, 0) = 2*base*0                when base <= 0.5
    #                    = 1 - 2*(1-base)*(1-0)   when base > 0.5
    #                    = 0                      when base <= 0.5
    #                    = 2*base - 1             when base > 0.5
    overlay_black = torch.where(
        image <= 0.5,
        torch.zeros_like(image),
        2.0 * image - 1.0,
    )

    return image * (1.0 - mask) + overlay_black * mask


def vhs_distortion(
    uv: torch.Tensor,
    strength: float = 0.04,
    bands: int = 18,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if uv.ndim < 3 or uv.shape[-3] != 2:
        raise ValueError("uv must have shape [..., 2, H, W]")

    strength = float(strength)
    bands = max(1, int(bands))

    leading = uv.shape[:-3]
    h = uv.shape[-2]
    device = uv.device
    dtype = uv.dtype

    u, v = uv.unbind(dim=-3)

    offsets = torch.rand(
        *leading,
        bands,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 2.0 - 1.0

    offsets = offsets * strength

    band_idx = torch.linspace(
        0,
        bands - 1,
        h,
        device=device,
    ).round().to(torch.long)

    offset_map = offsets[..., band_idx].unsqueeze(-1)

    return torch.stack((u + offset_map, v), dim=-3)


def stretch_smear(
    uv: torch.Tensor,
    amount: float = 0.2,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if uv.ndim < 3 or uv.shape[-3] != 2:
        raise ValueError("uv must have shape [..., 2, H, W]")

    h, w = uv.shape[-2:]
    num_strips = max(0, int(20 * float(amount)))
    max_stretch = max(10, int(w * 0.2))

    if num_strips == 0:
        return uv

    out = uv.clone()

    for _ in range(num_strips):
        y = int(torch.randint(0, h, (), device=uv.device, generator=generator).item())
        strip_h_max = max(1, int(h * 0.1))
        strip_h = int(torch.randint(1, strip_h_max + 1, (), device=uv.device, generator=generator).item())
        strip_h = min(strip_h, h - y)
        if strip_h <= 0:
            continue

        stretch = int(torch.randint(10, max_stretch + 1, (), device=uv.device, generator=generator).item())
        direction = 1 if int(torch.randint(0, 2, (), device=uv.device, generator=generator).item()) == 1 else -1
        factor = (w + stretch) / max(w, 1)

        cols = torch.arange(w, device=uv.device, dtype=uv.dtype)
        if direction > 0:
            src_cols = torch.floor(cols / factor).to(torch.long)
        else:
            src_cols = (w - 1) - torch.floor(((w - 1) - cols) / factor).to(torch.long)
        src_cols = src_cols.clamp(0, w - 1)

        out[..., :, y:y + strip_h, :] = out[..., :, y:y + strip_h, :].index_select(-1, src_cols)

    return out


def echo_trails(
    image: torch.Tensor,
    amount: float = 0.5,
    direction: float = 0.0,
) -> torch.Tensor:
    """
    Pixel-space echo trail shift.

    Args:
        image: Tensor of shape [..., C, H, W].
        amount: Normalized amount. 1.0 means max offset = 10% of image width.
        direction: Direction in radians.

    Returns:
        Shifted echo layer of shape [..., C, H, W].

    Notes:
        This returns only the shifted echo layer. Blend externally, e.g.:
            out = torch.lerp(image, echo_trails(image, amount, direction), 0.5)
    """
    if image.ndim < 3:
        raise ValueError("image must have shape [..., C, H, W]")

    h, w = image.shape[-2:]
    device = image.device

    max_offset = w * 0.1 * float(amount)

    dx = int(round(math.cos(float(direction)) * max_offset))
    dy = int(round(math.sin(float(direction)) * max_offset))

    # Positive dx/dy means the visible echo moves right/down.
    # Inverse sampling fills every output pixel by reading from the opposite direction.
    src_y = torch.arange(h, device=device) - dy
    src_x = torch.arange(w, device=device) - dx

    src_y = src_y.clamp(0, h - 1)
    src_x = src_x.clamp(0, w - 1)

    return image.index_select(-2, src_y).index_select(-1, src_x)


def echo_trail_uv(
    uv: torch.Tensor,
    amount: float = 0.5,
    direction: float = 0.0,
) -> torch.Tensor:
    if uv.ndim < 3 or uv.shape[-3] != 2:
        raise ValueError("uv must have shape [..., 2, H, W]")

    h, w = uv.shape[-2:]
    shift_px = max(w - 1, 1) * 0.1 * float(amount)
    du = math.cos(float(direction)) * shift_px / max(w - 1, 1)
    dv = math.sin(float(direction)) * shift_px / max(h - 1, 1)

    offset = torch.tensor((du, dv), device=uv.device, dtype=uv.dtype).view(*([1] * (uv.ndim - 3)), 2, 1, 1)
    return uv + offset


def pixelate(
    image: torch.Tensor,
    size: int = 8,
    intensity: float = 1.0,
    random_areas: float = 0.0,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Pixel-space pixelation matching the original parameters.

    Args:
        image: Tensor of shape [..., C, H, W].
        size: Pixel block size.
        intensity: Blend amount between original and pixelated image.
        random_areas: Probability of applying pixelation per block.
        generator: Optional torch random generator.

    Returns:
        Tensor of shape [..., C, H, W].
    """
    if image.ndim < 3:
        raise ValueError("image must have shape [..., C, H, W]")

    block = max(2, int(size))
    intensity = max(0.0, min(1.0, float(intensity)))
    random_areas = max(0.0, min(1.0, float(random_areas)))

    h, w = image.shape[-2:]
    device = image.device
    dtype = image.dtype

    y0 = (torch.arange(h, device=device) // block) * block
    x0 = (torch.arange(w, device=device) // block) * block
    yy, xx = torch.meshgrid(y0, x0, indexing="ij")

    pixelated = image[..., :, yy, xx]

    if random_areas > 0.0:
        rows = (h + block - 1) // block
        cols = (w + block - 1) // block

        block_mask = (
            torch.rand(
                *image.shape[:-3],
                1,
                rows,
                cols,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            < random_areas
        ).to(dtype)

        by = torch.arange(h, device=device) // block
        bx = torch.arange(w, device=device) // block
        block_mask = block_mask[..., :, by[:, None], bx[None, :]]

        target = torch.where(block_mask.bool(), pixelated, image)
    else:
        target = pixelated

    if intensity >= 1.0:
        return target

    return torch.lerp(image, target, intensity)


def pixelate_uv(
    uv: torch.Tensor,
    size: int = 8,
) -> torch.Tensor:
    if uv.ndim < 3 or uv.shape[-3] != 2:
        raise ValueError("uv must have shape [..., 2, H, W]")

    h, w = uv.shape[-2:]
    block = max(2, int(size))

    y0 = (torch.arange(h, device=uv.device) // block) * block
    x0 = (torch.arange(w, device=uv.device) // block) * block
    yy, xx = torch.meshgrid(y0, x0, indexing="ij")

    return uv[..., :, yy, xx]


def circuit_bend(
    x: torch.Tensor,
    chaos: float = 0.3,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError("x must have shape [..., C, H, W]")

    h, w = x.shape[-2:]
    num_glitches = max(0, int(h * w * float(chaos) * 0.001))
    max_offset = max(1, int(w * 0.1))

    if num_glitches == 0:
        return x

    out = x.clone()

    for _ in range(num_glitches):
        x0 = int(torch.randint(0, w, (), device=x.device, generator=generator).item())
        y0 = int(torch.randint(0, h, (), device=x.device, generator=generator).item())

        w_min = 1 if max_offset < 10 else 10
        block_w = int(torch.randint(w_min, max_offset + 1, (), device=x.device, generator=generator).item())
        block_h = int(torch.randint(1, 11, (), device=x.device, generator=generator).item())

        tx = (x0 + block_w) % w
        block_w = min(block_w, w - x0, w - tx)
        block_h = min(block_h, h - y0)
        if block_w <= 0 or block_h <= 0:
            continue

        a = out[..., :, y0:y0 + block_h, x0:x0 + block_w].clone()
        b = out[..., :, y0:y0 + block_h, tx:tx + block_w].clone()
        out[..., :, y0:y0 + block_h, x0:x0 + block_w] = b
        out[..., :, y0:y0 + block_h, tx:tx + block_w] = a

    return out


def posterize(
    image: torch.Tensor,
    intensity: int = 8,
) -> torch.Tensor:
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("image must have shape [..., 3, H, W]")

    levels = max(2, int(intensity))
    step = 1.0 / (levels - 1)
    return torch.round(image / step) * step


def corrupt_edges(
    image: torch.Tensor,
    amount: float = 0.5,
    density: float = 0.5,
    thickness: int = 5,
    intensity: float = 0.5,
    chaos: float = 0.5,
    invert_color: bool = False,
    pixelate: bool = False,
    rgb_split: bool = False,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("image must have shape [..., 3, H, W]")

    out = image.clone()
    source = image.clone()
    h, w = image.shape[-2:]

    edge_width = max(1, int(w * 0.15 * float(amount)))
    num_corruptions = max(0, int(10 + float(density) * 50))
    thickness = max(1, int(thickness))
    intensity = float(max(0.0, min(1.0, intensity)))
    chaos = float(max(0.0, chaos))

    for _ in range(num_corruptions):
        edge = int(torch.randint(0, 4, (), device=image.device, generator=generator).item())
        corruption_type = int(torch.randint(0, 5, (), device=image.device, generator=generator).item())

        if edge == 0:
            x1 = 0
            y1 = int(torch.randint(0, h, (), device=image.device, generator=generator).item())
            cw = int(edge_width * (0.5 + torch.rand((), device=image.device, generator=generator).item() * 0.5))
            ch = int(thickness * (1 + chaos * 5))
        elif edge == 1:
            cw = int(edge_width * (0.5 + torch.rand((), device=image.device, generator=generator).item() * 0.5))
            x1 = max(0, w - cw)
            y1 = int(torch.randint(0, h, (), device=image.device, generator=generator).item())
            ch = int(thickness * (1 + chaos * 5))
        elif edge == 2:
            x1 = int(torch.randint(0, w, (), device=image.device, generator=generator).item())
            y1 = 0
            cw = int(thickness * (1 + chaos * 5))
            ch = int(edge_width * (0.5 + torch.rand((), device=image.device, generator=generator).item() * 0.5))
        else:
            x1 = int(torch.randint(0, w, (), device=image.device, generator=generator).item())
            ch = int(edge_width * (0.5 + torch.rand((), device=image.device, generator=generator).item() * 0.5))
            y1 = max(0, h - ch)
            cw = int(thickness * (1 + chaos * 5))

        x2 = min(w, x1 + max(1, cw))
        y2 = min(h, y1 + max(1, ch))
        cw = x2 - x1
        ch = y2 - y1
        if cw <= 0 or ch <= 0:
            continue

        if corruption_type == 0:
            shift = int(cw * chaos * 2) - cw
            dst_x1 = max(0, x1 + shift)
            dst_x2 = min(w, x2 + shift)
            if dst_x2 > dst_x1:
                src_x1 = x1 + (dst_x1 - (x1 + shift))
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                out[..., :, y1:y2, dst_x1:dst_x2] = source[..., :, y1:y2, src_x1:src_x2]

        elif corruption_type == 1:
            region = out[..., :, y1:y2, x1:x2]
            if rgb_split:
                offset = max(1, int(chaos * 10))
                xs = torch.arange(x1, x2, device=image.device)
                rx = (xs + offset).clamp(0, w - 1)
                gx = (xs - offset).clamp(0, w - 1)
                region_r = out[..., 0:1, y1:y2, :].index_select(-1, rx)
                region_g = out[..., 1:2, y1:y2, :].index_select(-1, gx)
                out[..., 0:1, y1:y2, x1:x2] = region_r
                out[..., 1:2, y1:y2, x1:x2] = region_g
            elif invert_color:
                out[..., :, y1:y2, x1:x2] = 1.0 - region
            else:
                rand_mask = (torch.rand(*region.shape[:-3], 1, ch, cw, device=image.device, dtype=image.dtype, generator=generator) < intensity)
                corrupt = torch.randint(0, 2, region.shape, device=image.device, generator=generator).to(image.dtype)
                out[..., :, y1:y2, x1:x2] = torch.where(rand_mask, corrupt, region)

        elif corruption_type == 2:
            if pixelate:
                px = max(2, thickness // 2)
                y0 = (torch.arange(ch, device=image.device) // px) * px
                x0 = (torch.arange(cw, device=image.device) // px) * px
                yy, xx = torch.meshgrid(y0, x0, indexing="ij")
                block = out[..., :, y1:y2, x1:x2]
                out[..., :, y1:y2, x1:x2] = block[..., :, yy, xx]

        elif corruption_type == 3:
            num_lines = max(1, int(3 + chaos * 7))
            line_h = max(1, ch // num_lines)
            for line in range(num_lines):
                ly = y1 + line * line_h
                lh = min(line_h, y2 - ly)
                if lh <= 0:
                    continue
                line_shift = int((torch.rand((), device=image.device, generator=generator).item() - 0.5) * cw * 2 * chaos)
                dst_x1 = max(0, x1 + line_shift)
                dst_x2 = min(w, x2 + line_shift)
                if dst_x2 <= dst_x1:
                    continue
                src_x1 = x1 + (dst_x1 - (x1 + line_shift))
                src_x2 = src_x1 + (dst_x2 - dst_x1)
                out[..., :, ly:ly + lh, dst_x1:dst_x2] = source[..., :, ly:ly + lh, src_x1:src_x2]

        else:
            region = out[..., :, y1:y2, x1:x2]
            active = (torch.rand(*region.shape[:-3], 1, ch, cw, device=image.device, dtype=image.dtype, generator=generator) < (intensity * 0.7))
            complete = (torch.rand(*region.shape[:-3], 1, ch, cw, device=image.device, dtype=image.dtype, generator=generator) < 0.3)
            random_rgb = torch.rand(region.shape, device=image.device, dtype=image.dtype, generator=generator)

            q = (region.clamp(0.0, 1.0) * 255.0).round().to(torch.int16)
            bits = torch.randint(0, 8, q.shape, device=image.device, generator=generator, dtype=torch.int16)
            flips = torch.bitwise_left_shift(torch.ones_like(bits), bits)
            q_flip = torch.bitwise_xor(q, flips).clamp(0, 255)
            flipped = q_flip.to(image.dtype) / 255.0

            corrupted = torch.where(complete, random_rgb, flipped)
            out[..., :, y1:y2, x1:x2] = torch.where(active, corrupted, region)

    if chaos > 0.3:
        num_horizontal_glitches = max(0, int(chaos * 5))
        for _ in range(num_horizontal_glitches):
            y = int(torch.randint(0, h, (), device=image.device, generator=generator).item())
            gh = max(1, int(thickness * (0.5 + torch.rand((), device=image.device, generator=generator).item())))
            gh = min(gh, h - y)
            if gh <= 0:
                continue
            offset = int((torch.rand((), device=image.device, generator=generator).item() - 0.5) * w * 0.2 * chaos)
            dst_x1 = max(0, offset)
            dst_x2 = min(w, w + offset)
            src_x1 = max(0, -offset)
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            if dst_x2 > dst_x1:
                out[..., :, y:y + gh, dst_x1:dst_x2] = source[..., :, y:y + gh, src_x1:src_x2]

    return out.clamp(0.0, 1.0)


def rgb_spatial_shift(
    image: torch.Tensor,
    red_offset: tuple[int, int] = (0, 0),
    green_offset: tuple[int, int] = (0, 0),
    blue_offset: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """
    RGB channel spatial shift in pixel space.

    Args:
        image: Tensor of shape [..., 3, H, W].
        red_offset: (dx, dy) pixel offset for the red channel.
        green_offset: (dx, dy) pixel offset for the green channel.
        blue_offset: (dx, dy) pixel offset for the blue channel.

    Returns:
        Tensor of shape [..., 3, H, W].

    Notes:
        Uses inverse sampling with clamp-to-edge fill, matching the original
        canvas implementation's clampPixel behavior.
    """
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError("image must have shape [..., 3, H, W]")

    h, w = image.shape[-2:]
    device = image.device

    def shift_channel(channel: torch.Tensor, offset: tuple[int, int]) -> torch.Tensor:
        dx, dy = int(offset[0]), int(offset[1])

        # Inverse sampling:
        # output[y, x] reads input[y + dy, x + dx].
        # This matches the original source-read convention:
        # data[x] = tempData[x + offset].
        src_y = torch.arange(h, device=device) + dy
        src_x = torch.arange(w, device=device) + dx

        src_y = src_y.clamp(0, h - 1)
        src_x = src_x.clamp(0, w - 1)

        return channel.index_select(-2, src_y).index_select(-1, src_x)

    r = shift_channel(image[..., 0, :, :], red_offset)
    g = shift_channel(image[..., 1, :, :], green_offset)
    b = shift_channel(image[..., 2, :, :], blue_offset)

    return torch.stack((r, g, b), dim=-3)
