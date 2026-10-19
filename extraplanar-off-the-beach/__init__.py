import torch
import torch.nn.functional as F
import math
from tensor_image import save_image
from tensor_image.effects import dilate2d, make_hsv, hsv_to_rgb, oklch_to_rgb, scanlines, vhs_distortion, corrupt_edges, circuit_bend, echo_trail_uv, rgb_spatial_shift, echo_trails, pixelate

height = 4096
width = 4096
device = "cuda:0"
dtype = torch.float32
r, g, b = 0, 1, 2


def update_image(image: torch.Tensor) -> None:
    c, h, w = image.shape
    u_quant_levels = 34
    generator = torch.Generator(device=device)
    uv = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, u_quant_levels, device=device, dtype=dtype),
        torch.linspace(0, 1, u_quant_levels, device=device, dtype=dtype),
        indexing="ij"
    ), dim=0).flip(0)

    uv2 = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij"
    ), dim=0).flip(0)
    uv2 = vhs_distortion(uv2, strength=0.01, bands=8, generator=generator)

    generator.manual_seed(0)
    # uv = vhs_distortion(uv, strength=0.04, bands=14, generator=generator)

    # uv = echo_trail_uv(uv, amount=0.01, direction=0.32 * torch.pi * 2)

    pos = 0.5, 0.5
    scale = 0.75, 0.75
    brightness = 2.0
    saturation = -0.1
    u, v = uv
    u_d, v_d = (u - pos[0]) / scale[0], (v - pos[1]) / scale[1]

    # u = 1 - u

    circle = (u_d**2 + v_d**2).sqrt()

    theta = torch.atan2(v_d, u_d)
    t = theta + torch.pi
    hue = t / (2 * torch.pi) - 3/12

    # inner = 0.00
    # outer = 0.16
    # sat = ((circle - inner) / (outer - inner)).clamp(0, 1)
    # sat = sat * sat * (3 - 2 * sat)   # smoothstep
    sat = 1

    hsv = make_hsv(hue, sat, sat)
    rgb = hsv_to_rgb(hsv)

    blend = (circle / 0.32).clamp(0, 1)
    blend = blend * blend * (3 - 2 * blend)

    center_color = torch.tensor([0.0, 0.0, 0.6], device=device)[:, None, None]  # white
    rgb_blend = center_color * (1 - blend) + rgb * blend

    quant_levels = 16
    rgb_quant = (rgb_blend * quant_levels).floor() / quant_levels

    height = 0.02 + 2*(circle / 1.2).clamp(0, 1) ** 2
    height += torch.randn_like(height, generator=generator).abs()**3 * 0.04
    # height = 0.5 + 0.5 * torch.cos(theta * 6)

    # height_levels = 32
    # height = (height * height_levels).floor() / height_levels

    render_isometric_blocks(
        rgb_quant, height, image,
        pitch_deg=45,
        zoom=256,
        gap=0.3,
        background=(0.0, 0.0, 0.0),
        uv_map=uv2,
    )

    noise = torch.randn_like(image) * 0.2
    noise_levels = 4
    noise = (noise * noise_levels).floor() / noise_levels
    image[:] += noise

    image2 = image.clone()
    add_gap_glow(
        image2,
        glow_color=(0.6, 0.6, 1.0),
        glow_strength=2.0,
        bloom_radius=64,
        bloom_sigma=32,
        core_strength=1.0,
    )
    bent_glow = circuit_bend(image2 - image, chaos=0.05, generator=generator)
    bent_glow2 = circuit_bend(image2 - image, chaos=0.2, generator=generator)

    def do_bend_glow(a):
        b = -a
        a = a * 0.75
        return (image2 - image) * (1 - a - b) + bent_glow * a + bent_glow2*b
    glow = do_bend_glow(0.75)
    image[:] = image*0.4*(uv2[0] + uv2[1].flip(0))**(1.5) + glow

    # generator.manual_seed(0)
    # image[:] = image*0.0 + corrupt_edges(image, amount=1.0, density=0.5, thickness=64, intensity=0.5, chaos=0, generator=generator)*1.0

    # generator.manual_seed(0)
    # image[:] = image*0.0 + corrupt_edges(image, amount=1.0, density=0.0, thickness=32, intensity=0.0, chaos=1.0, generator=generator)*1.0

    image[:] = (image+saturation)*brightness
    # image[:] = rgb


def gaussian_kernel1d(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    return k


def blur_image(image: torch.Tensor, radius: int = 16, sigma: float = 8.0) -> torch.Tensor:
    """
    image: [C, H, W]
    returns blurred image of same shape
    """
    c, h, w = image.shape
    device = image.device
    dtype = image.dtype

    k1d = gaussian_kernel1d(radius, sigma, device, dtype)

    kx = k1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    x = image.unsqueeze(0)  # [1, C, H, W]

    x = F.conv2d(x, kx, padding=(0, radius), groups=c)
    x = F.conv2d(x, ky, padding=(radius, 0), groups=c)

    return x[0]


def add_gap_glow(
    image: torch.Tensor,
    glow_color=(1.0, 0.0, 1.0),
    glow_strength: float = 1.0,
    bloom_radius: int = 24,
    bloom_sigma: float = 10.0,
    core_strength: float = 0.5,
) -> None:
    """
    image: [3, H, W]
    Modifies image in place.
    Assumes background is black and gaps remain black.
    """
    device = image.device
    dtype = image.dtype

    # Pixels where nothing was drawn
    gap_mask = (image.amax(dim=0, keepdim=True) <= 1e-6).to(dtype)   # [1, H, W]

    # Bright core in the gap
    glow_rgb = torch.tensor(glow_color, device=device, dtype=dtype)[:, None, None]
    core = gap_mask * glow_rgb * core_strength

    # Bloom spreading outward
    bloom = blur_image(gap_mask.repeat(3, 1, 1), radius=bloom_radius, sigma=bloom_sigma)
    bloom = bloom * glow_rgb * glow_strength

    # Add both
    image[:] = (image + core + bloom).clamp(0.0, 1.0)


def render_isometric_blocks(
    rgb_grid: torch.Tensor,              # [3, gh, gw], values in [0, 1]
    height_grid: torch.Tensor,           # [gh, gw]
    image: torch.Tensor,                 # [3, out_h, out_w]
    yaw_deg: float = 225.0,              # rotation around world Z
    pitch_deg: float = 35.264,           # tilt downward; iso-ish default
    zoom: float = 64.0,                  # screen scale
    height_scale: float = 6.0,           # multiplies height_grid values
    tile_size: float = 1.0,              # world-space tile width/depth
    gap: float = 0.2,                    # 0 = touching, 0.1 = 10% gap
    origin: tuple[float, float] | None = None,   # screen-space center
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    side_shades: tuple[float, float] = (0.70, 0.50),  # x-side, y-side
    uv_map: torch.Tensor | None = None,  # [2, out_h, out_w]
) -> torch.Tensor:
    """
    Render a grid of colored height-extruded boxes using an orthographic
    isometric-style projection.

    World axes:
      x -> grid columns
      y -> grid rows
      z -> height

    Returns:
      image: [3, out_h, out_w]
    """
    assert rgb_grid.ndim == 3 and rgb_grid.shape[0] == 3, "rgb_grid must be [3, gh, gw]"
    assert height_grid.ndim == 2, "height_grid must be [gh, gw]"
    assert rgb_grid.shape[1:] == height_grid.shape, "rgb_grid and height_grid spatial sizes must match"

    device = image.device
    dtype = image.dtype

    gh, gw = height_grid.shape
    _, out_h, out_w = image.shape

    pad_x = pad_y = 0

    if uv_map is not None:
        assert uv_map.shape == (2, out_h, out_w)

        xs = torch.arange(out_w, device=device, dtype=dtype)[None, :]
        ys = torch.arange(out_h, device=device, dtype=dtype)[:, None]

        dx = uv_map[0] * (out_w - 1) - xs
        dy = uv_map[1] * (out_h - 1) - ys

        pad_x = math.ceil(dx.abs().max().item()) + 1
        pad_y = math.ceil(dy.abs().max().item()) + 1

    if origin is None:
        origin = ((out_w - 1) * 0.5, (out_h - 1) * 0.5)

    bg = torch.tensor(background, device=device, dtype=dtype)[:, None, None]
    image[:] = bg

    z_buffer = torch.full(
        (out_h, out_w),
        -torch.inf,
        device=device,
        dtype=dtype,
    )

    # --- camera rotation: world -> camera ---
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Rotate around Z, then around X
    Rz = torch.tensor([
        [ cy, -sy, 0.0],
        [ sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, sp,  cp],
        [0.0, -cp, sp],
    ], device=device, dtype=dtype)

    R = Rx @ Rz

    screen_center = torch.tensor(origin, device=device, dtype=dtype)

    def project(points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        points_3d: [N, 3]
        returns:
          screen_xy: [N, 2]
          depth:     [N]
        """
        cam = points_3d @ R.T
        screen = torch.empty((points_3d.shape[0], 2), device=device, dtype=dtype)
        screen[:, 0] = screen_center[0] + cam[:, 0] * zoom
        screen[:, 1] = screen_center[1] - cam[:, 1] * zoom
        depth = cam[:, 2]
        return screen, depth

    def polygon_area_2d(pts: torch.Tensor) -> torch.Tensor:
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * torch.sum(x * torch.roll(y, shifts=-1) - torch.roll(x, shifts=-1) * y)

    def rasterize_convex_polygon(pts_2d: torch.Tensor, depth_value: torch.Tensor, color: torch.Tensor) -> None:
        """
        pts_2d: [N, 2] convex polygon in screen space
        depth_value: scalar
        color: [3]
        """
        min_x = max(int(torch.floor(pts_2d[:, 0].min()).item()) - pad_x, 0)
        max_x = min(int(torch.ceil(pts_2d[:, 0].max()).item()) + pad_x, out_w - 1)
        min_y = max(int(torch.floor(pts_2d[:, 1].min()).item()) - pad_y, 0)
        max_y = min(int(torch.ceil(pts_2d[:, 1].max()).item()) + pad_y, out_h - 1)

        if min_x > max_x or min_y > max_y:
            return

        xs = torch.arange(min_x, max_x + 1, device=device, dtype=dtype) + 0.5
        ys = torch.arange(min_y, max_y + 1, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        if uv_map is not None:
            sample_x = uv_map[0, min_y:max_y + 1, min_x:max_x + 1] * (out_w - 1) + 0.5
            sample_y = uv_map[1, min_y:max_y + 1, min_x:max_x + 1] * (out_h - 1) + 0.5
        else:
            sample_x = xx
            sample_y = yy

        area = polygon_area_2d(pts_2d)
        sign = 1.0 if area.item() >= 0 else -1.0

        inside = torch.ones_like(xx, dtype=torch.bool)

        n = pts_2d.shape[0]
        eps = 1e-6
        for i in range(n):
            a = pts_2d[i]
            b = pts_2d[(i + 1) % n]
            edge = b - a
            cross = edge[0] * (sample_y - a[1]) - edge[1] * (sample_x - a[0])
            inside &= (sign * cross >= -eps)

        if not inside.any():
            return

        z_sub = z_buffer[min_y:max_y + 1, min_x:max_x + 1]
        visible = inside & (depth_value > z_sub)

        if not visible.any():
            return

        z_sub[visible] = depth_value
        for ch in range(3):
            img_sub = image[ch, min_y:max_y + 1, min_x:max_x + 1]
            img_sub[visible] = color[ch]

    def draw_face(face_3d: torch.Tensor, color: torch.Tensor) -> None:
        pts_2d, depths = project(face_3d)
        depth_value = depths.mean()
        rasterize_convex_polygon(pts_2d, depth_value, color)

    # --- render tiles ---
    # We still use a painter-ish iteration order to reduce weird overlaps,
    # but the z-buffer does the main visibility work.
    tile_records = []
    for gy in range(gh):
        for gx in range(gw):
            h_val = float(height_grid[gy, gx].item())
            x_center = (gx - (gw - 1) * 0.5) * tile_size
            y_center = (gy - (gh - 1) * 0.5) * tile_size
            top_center = torch.tensor([[x_center, y_center, h_val * height_scale]],
                                      device=device, dtype=dtype)
            _, d = project(top_center)
            tile_records.append((float(d.item()), gy, gx))

    # far -> near
    tile_records.sort(key=lambda t: t[0])

    for _, gy, gx in tile_records:
        color = rgb_grid[:, gy, gx].clamp(0.0, 1.0)
        h_val = float(height_grid[gy, gx].item()) * height_scale
        h_val = max(0.0, h_val)

        x = (gx - (gw - 1) * 0.5) * tile_size
        y = (gy - (gh - 1) * 0.5) * tile_size

        half_size = tile_size * (1.0 - gap) * 0.5
        x0 = x - half_size
        x1 = x + half_size
        y0 = y - half_size
        y1 = y + half_size
        z0 = 0.0
        z1 = h_val

        # Top face
        top = torch.tensor([
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], device=device, dtype=dtype)

        # Determine which x-side is facing the camera
        x_face_centers = torch.tensor([
            [x0, y,  z1 * 0.5],
            [x1, y,  z1 * 0.5],
        ], device=device, dtype=dtype)
        _, x_face_depths = project(x_face_centers)

        if x_face_depths[1] >= x_face_depths[0]:
            x_side = torch.tensor([
                [x1, y0, z0],
                [x1, y1, z0],
                [x1, y1, z1],
                [x1, y0, z1],
            ], device=device, dtype=dtype)
        else:
            x_side = torch.tensor([
                [x0, y1, z0],
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z1],
            ], device=device, dtype=dtype)

        # Determine which y-side is facing the camera
        y_face_centers = torch.tensor([
            [x,  y0, z1 * 0.5],
            [x,  y1, z1 * 0.5],
        ], device=device, dtype=dtype)
        _, y_face_depths = project(y_face_centers)

        if y_face_depths[1] >= y_face_depths[0]:
            y_side = torch.tensor([
                [x0, y1, z0],
                [x1, y1, z0],
                [x1, y1, z1],
                [x0, y1, z1],
            ], device=device, dtype=dtype)
        else:
            y_side = torch.tensor([
                [x1, y0, z0],
                [x0, y0, z0],
                [x0, y0, z1],
                [x1, y0, z1],
            ], device=device, dtype=dtype)

        # Draw sides first, then top
        if z1 > 0.0:
            draw_face(x_side, (color * side_shades[0]).clamp(0.0, 1.0))
            draw_face(y_side, (color * side_shades[1]).clamp(0.0, 1.0))
        draw_face(top, color)

    return image


def main() -> None:
    image = torch.zeros(3, height, width, device=device)

    update_image(image)

    save_image(
        resize_image(image, (height//4, width//4)),
        "out.png",
        value_range=(0.0, 1.0),
    )


def resize_image(
    image: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0].clamp(0, 1)


if __name__ == "__main__":
    main()
