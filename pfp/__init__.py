import torch
from tensor_image import save_image
from tensor_image.effects import dilate2d, make_hsv, hsv_to_rgb, oklch_to_rgb, scanlines, vhs_distortion, corrupt_edges, circuit_bend, echo_trail_uv, rgb_spatial_shift, echo_trails, pixelate

height = 4096
width = 4096
device = "cuda:0"
dtype = torch.float32


def update_image(image: torch.Tensor) -> None:
    c, h, w = image.shape
    generator = torch.Generator(device=device)
    uv = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij"
    ), dim=0).flip(0)

    generator.manual_seed(4)
    uv = vhs_distortion(uv, strength=0.04, bands=14, generator=generator)

    # uv = echo_trail_uv(uv, amount=0.01, direction=0.32 * torch.pi * 2)

    u, v = uv
    # u = 1 - u
    u += 0.025

    pos = 0.5, 0.5
    scale = 0.75, 0.75
    brightness = 0
    contrast = 1
    rgb_offsets = tuple(x / w for x in (4, 10, 46))

    u_d, v_d = (u - pos[0]) / scale[0], (v - pos[1]) / scale[1]

    circle3_uv_scale = 7.5

    circle3 = (u_d * circle3_uv_scale)**2 + (v_d * circle3_uv_scale)**2
    # circle2 = ((u_d * 24)**2 + (v_d * 24)**2).clamp_max(1)
    circle1 = u_d**2 + v_d**2
    circle_lin = circle1.sqrt()

    theta = torch.atan2(-v_d, u_d)

    t = theta + circle_lin*torch.pi
    h = t / (2 * torch.pi) - 7/12
    hsv = make_hsv(h, 1, 1)
    rgb = hsv_to_rgb(hsv)
    # rgb = torch.lerp(rgb, oklch_to_rgb(0.25, 1, h), 1)

    wiggle_parts = []
    circle4_lin = (((u_d / 2)**2)**2 + (v_d / 2**2)**2).sqrt()
    wiggle = torch.zeros_like(theta)
    for j in range(512):
        wiggle += (torch.cos((theta + circle4_lin)*((j+1)*4096)))
    wiggle = (wiggle - wiggle.min()) / (wiggle.max() - wiggle.min())
    wiggle *= 2
    wiggle += dilate2d(wiggle, iterations=4)
    wiggle *= 0.5

    blob = (0.05 / (circle3**2)).clip(0,1)

    noisy_rgb = (rgb * (wiggle - 0)) * (1 - circle_lin*1.25)
    rgb_sat = 3
    saturated_rgb = (noisy_rgb*rgb_sat).clip(0, 1.5)/rgb_sat

    black_hole = (saturated_rgb*0.7 + blob*4).clip(0, 1)

    black_hole = rgb_spatial_shift(black_hole, (rgb_offsets[0], 0), (rgb_offsets[1], 0), (rgb_offsets[2], 0))

    black_hole = torch.lerp(black_hole, scanlines(black_hole, spacing=9.33, thickness=1.54), 0.25)

    generator.manual_seed(0)
    black_hole = pixelate(black_hole, size=7.7, intensity=1, random_areas=0.24, generator=generator)

    black_hole = torch.lerp(black_hole, echo_trails(black_hole, amount=0.01, direction=0.32*torch.pi*2), 0.5)

    generator.manual_seed(8)
    black_hole = corrupt_edges(black_hole, amount=1, density=0.5, thickness=13.92, intensity=0.6, chaos=1, invert_color=True, pixelate=True, generator=generator)

    generator.manual_seed(7)
    black_hole = circuit_bend(black_hole, chaos=0.04, generator=generator)

    generator.manual_seed(10)
    black_hole = corrupt_edges(black_hole, amount=1, density=0.5, thickness=13.92, intensity=0.6, chaos=1, invert_color=True, pixelate=True, generator=generator)

    black_hole = 1 - black_hole

    image[:] = black_hole
    # image[:] = rgb


def main() -> None:
    image = torch.zeros(3, height, width, device=device)

    update_image(image)

    save_image(
        image,
        "out.png",
        value_range=(0.0, 1.0),
    )


if __name__ == "__main__":
    main()
