import torch
from tensor_image import save_image
from effects import dilate2d, make_hsv, hsv_to_rgb, hsv_to_r, hsv_to_g, hsv_to_b

height = 4096
width = 4096
device = "cuda:0"
dtype = torch.float32


def update_image(image: torch.Tensor) -> None:
    c, h, w = image.shape
    uv = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij"
    ), dim=0).flip(0)
    u, v = uv

    pos = 0.5, 0.5
    scale = 0.75, 0.75
    brightness = 0
    contrast = 1

    u_d, v_d = (u - pos[0]) / scale[0], (v - pos[1]) / scale[1]

    circle3_uv_scale = 8

    circle6 = (u_d * 6)**2 + (v_d * 6)**2
    circle3 = (u_d * circle3_uv_scale)**2 + (v_d * circle3_uv_scale)**2
    # circle2 = ((u_d * 24)**2 + (v_d * 24)**2).clamp_max(1)
    circle1 = (u_d)**2 + (v_d)**2
    circle_lin = circle1.sqrt()
    circle4_lin = ((u_d / 2**2)**2 + (v_d / 2**2)**2).sqrt()

    theta = torch.atan2(-v_d, u_d)
    t = theta + circle_lin*torch.pi
    h = t / (2 * torch.pi) - 7/12
    hsv = make_hsv(h, 1, 1)
    rgb = hsv_to_rgb(hsv)

    wiggle = torch.zeros_like(theta)
    for i in range(512):
        wiggle += (torch.cos((theta + circle4_lin)*((i+1)*4096)))
    wiggle /= wiggle.abs().max()
    wiggle = (wiggle - wiggle.min()) / (wiggle.max() - wiggle.min())
    wiggle *= 2
    wiggle += dilate2d(wiggle, iterations=4)
    # wiggle += (1-dilate2d(1-dilate2d(wiggle, iterations=4), iterations=4))*2
    wiggle *= 0.5

    wheel = 0.04 / (circle1 + 0.21)**2
    # wheel = 0.01 / (circle1 + 0.1)**2 * 1.25 + 0.1
    rgb_wheel = rgb * wheel
    blob = (0.05 / (circle3**2)).clip(0,1)
    # weight = 1 - (((1 - circle6) * 0.1 + 0.8) + wiggle)

    # image[:] = (torch.lerp(rgb_wheel, blob, weight) + rgb_wheel)*rgb_wheel
    noisy_rgb = (rgb * (wiggle - 0)) * (1 - circle1.sqrt()*1.25)
    rgb_sat = 3
    saturated_rgb = (noisy_rgb*rgb_sat).clip(0, 1.5)/rgb_sat

    image[:] = (saturated_rgb*0.7 + blob*4).clip(0, 1)

    image[:] = image * contrast + brightness


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
