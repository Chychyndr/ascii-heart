# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "glfw>=2.8.0",
#     "PyOpenGL>=3.1.7",
#     "Pillow>=10.0.0",
#     "numpy>=1.26.0",
# ]
# ///

import math
import time
from pathlib import Path

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont


GRID_W = 220
GRID_H = 100
FONT_SIZE = 9

ROTATION_SPEED = 1.45
INTRO_DURATION = 1.8
BEAT_PERIOD = 1.15

THETA_STEPS = 360
PHI_STEPS = 820
BISECTION_STEPS = 40

CAMERA_DISTANCE = 5.6
SCREEN_SCALE_X = 114.0
SCREEN_SCALE_Y = 84.0

# Форма: x = ширина, y = глубина, z = высота.
HEART_SHAPE_SCALE = np.array([1.85, 1.38, 0.94], dtype=np.float32)

ASCII_RAMP = (
    "   ...'''```^^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao"
    "*#MW&8%B@$"
)
RAMP_ARRAY = np.array(list(ASCII_RAMP), dtype="<U1")

LIGHT = np.array([-0.35, 0.70, 1.0], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)

VIEW = np.array([0.0, 1.0, 0.0], dtype=np.float32)
HALF_VECTOR = LIGHT + VIEW
HALF_VECTOR /= np.linalg.norm(HALF_VECTOR)

GRID_CELLS = GRID_W * GRID_H
NEG_INF = np.float32(-1.0e20)


class FontData:
    def __init__(self, font, char_w: int, char_h: int, atlas: dict[str, np.ndarray]):
        self.font = font
        self.char_w = char_w
        self.char_h = char_h
        self.atlas = atlas


class SurfaceData:
    def __init__(self, points: np.ndarray, normals: np.ndarray):
        self.x = points[:, 0]
        self.y = points[:, 1]
        self.z = points[:, 2]
        self.nx = normals[:, 0]
        self.ny = normals[:, 1]
        self.nz = normals[:, 2]
        self.count = len(points)


class FrameData:
    def __init__(self):
        self.char_buffer = np.full((GRID_H, GRID_W), " ", dtype="<U1")
        self.heart_mask = np.zeros((GRID_H, GRID_W), dtype=bool)
        self.brightness = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.height = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.shadow = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.nearest = np.full(GRID_CELLS, NEG_INF, dtype=np.float32)

    def clear(self):
        self.char_buffer.fill(" ")
        self.heart_mask.fill(False)
        self.brightness.fill(0.0)
        self.height.fill(0.0)
        self.shadow.fill(0.0)
        self.nearest.fill(NEG_INF)


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def ease_out_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_smooth(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_out_back(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (x - 1.0) ** 3 + c1 * (x - 1.0) ** 2


def heartbeat_scale(t: float) -> float:
    x = (t % BEAT_PERIOD) / BEAT_PERIOD

    first_hit = 0.085 * math.exp(-((x - 0.12) / 0.045) ** 2)
    second_hit = 0.050 * math.exp(-((x - 0.23) / 0.030) ** 2)
    settle = 0.018 * math.exp(-((x - 0.34) / 0.060) ** 2)

    return 1.0 + first_hit + second_hit - settle


def load_monospace_font(size: int):
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
    ]

    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


def build_font_data(size: int) -> FontData:
    font = load_monospace_font(size)

    try:
        char_w = int(math.ceil(font.getlength("M")))
    except Exception:
        bbox = font.getbbox("M")
        char_w = bbox[2] - bbox[0]

    char_w = max(1, char_w)

    bbox = font.getbbox("Mg")
    char_h = max(1, bbox[3] - bbox[1] + 3)
    y_offset = -bbox[1]

    chars = sorted(set(ASCII_RAMP) | {";", ":", "."})
    atlas: dict[str, np.ndarray] = {}

    for ch in chars:
        if ch == " ":
            continue

        img = Image.new("L", (char_w, char_h), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, y_offset), ch, font=font, fill=255)
        atlas[ch] = np.asarray(img, dtype=np.float32) / 255.0

    return FontData(font, char_w, char_h, atlas)


def heart_function(x, y, z):
    a = x * x + 2.25 * y * y + z * z - 1.0
    return a * a * a - x * x * z * z * z - 0.1125 * y * y * z * z * z


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1.0e-8)


def heart_gradient(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    a = x * x + 2.25 * y * y + z * z - 1.0

    gx = 6.0 * x * a * a - 2.0 * x * z * z * z
    gy = 13.5 * y * a * a - 0.225 * y * z * z * z
    gz = 6.0 * z * a * a - 3.0 * x * x * z * z - 0.3375 * y * y * z * z

    normals = np.stack([gx, gy, gz], axis=1)
    return normalize_rows(normals).astype(np.float32)


def make_heart_surface() -> SurfaceData:
    theta = np.linspace(0.0, math.pi, THETA_STEPS, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * math.pi, PHI_STEPS, endpoint=False, dtype=np.float32)

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    sin_theta = np.sin(theta_grid)

    directions = np.stack(
        [
            sin_theta * np.cos(phi_grid),
            sin_theta * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    lo = np.zeros(len(directions), dtype=np.float32)
    hi = np.full(len(directions), 2.6, dtype=np.float32)

    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) * 0.5
        p = directions * mid[:, None]
        inside = heart_function(p[:, 0], p[:, 1], p[:, 2]) < 0.0

        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)

    radius = (lo + hi) * 0.5
    points = directions * radius[:, None]
    normals = heart_gradient(points)

    points -= (points.min(axis=0) + points.max(axis=0)) * 0.5

    points *= HEART_SHAPE_SCALE

    # После неравномерного масштаба нормали нужно пересчитать через inverse transpose.
    normals = normalize_rows(normals / HEART_SHAPE_SCALE).astype(np.float32)

    return SurfaceData(points.astype(np.float32), normals)


def build_shadow_template() -> np.ndarray:
    y_grid, x_grid = np.mgrid[0:GRID_H, 0:GRID_W]

    shadow_y = GRID_H - 9
    dx = (x_grid - GRID_W * 0.5) / (GRID_W * 0.5)
    dy = (y_grid - shadow_y) / 4.8

    value = np.exp(-(dx * dx * 11.0 + dy * dy * 2.6)).astype(np.float32)
    value[value <= 0.16] = 0.0

    return value


def render_ascii(surface: SurfaceData, frame: FrameData, shadow_template: np.ndarray, t: float):
    frame.clear()

    intro = clamp(t / INTRO_DURATION, 0.0, 1.0)
    intro_scale = 0.30 + 0.70 * ease_out_back(intro)
    intro_alpha = ease_in_out_smooth(intro)
    intro_shift = -0.30 * (1.0 - ease_out_cubic(intro))

    model_scale = intro_scale * heartbeat_scale(t)
    vertical_shift = intro_shift + 0.012 * math.sin(t * 2.0)
    angle = t * ROTATION_SPEED

    c = math.cos(angle)
    s = math.sin(angle)

    x = surface.x * model_scale
    y = surface.y * model_scale
    z = surface.z * model_scale + vertical_shift

    px = x * c - y * s
    depth = x * s + y * c
    pz = z

    nx = surface.nx * c - surface.ny * s
    ny = surface.nx * s + surface.ny * c
    nz = surface.nz

    denom = CAMERA_DISTANCE - depth
    visible = denom > 0.05

    perspective = np.empty_like(depth, dtype=np.float32)
    perspective[visible] = 1.0 / denom[visible]

    screen_x = (px * perspective * SCREEN_SCALE_X + GRID_W * 0.5).astype(np.int32)
    screen_y = (-pz * perspective * SCREEN_SCALE_Y + GRID_H * 0.5 - 2.0).astype(np.int32)

    visible &= (
        (screen_x >= 0)
        & (screen_x < GRID_W)
        & (screen_y >= 0)
        & (screen_y < GRID_H)
    )

    if not np.any(visible):
        return intro_alpha

    sx = screen_x[visible]
    sy = screen_y[visible]
    d = depth[visible]
    vx = nx[visible]
    vy = ny[visible]
    vz = nz[visible]
    hz = pz[visible]

    cells = sy * GRID_W + sx

    np.maximum.at(frame.nearest, cells, d)
    front = d >= frame.nearest[cells] - 1.0e-6

    sx = sx[front]
    sy = sy[front]
    d = d[front]
    vx = vx[front]
    vy = vy[front]
    vz = vz[front]
    hz = hz[front]
    cells = cells[front]

    diffuse = np.clip(vx * LIGHT[0] + vy * LIGHT[1] + vz * LIGHT[2], 0.0, 1.0)
    specular = np.clip(vx * HALF_VECTOR[0] + vy * HALF_VECTOR[1] + vz * HALF_VECTOR[2], 0.0, 1.0) ** 32.0
    facing = np.clip(vy * 0.5 + 0.5, 0.0, 1.0)
    rim = (1.0 - np.abs(np.clip(vy, -1.0, 1.0))) ** 1.45

    d_min = float(d.min())
    d_max = float(d.max())
    depth_norm = (d - d_min) / (d_max - d_min + 1.0e-6)

    z_min = float(hz.min())
    z_max = float(hz.max())
    height_norm = (hz - z_min) / (z_max - z_min + 1.0e-6)

    brightness = (
        0.08
        + diffuse * 0.52
        + specular * 0.38
        + rim * 0.18
        + depth_norm * 0.12
    )
    brightness *= 0.58 + facing * 0.42
    brightness = np.clip(brightness, 0.0, 1.0)

    counts = np.bincount((sy * GRID_W + sx), minlength=GRID_CELLS)
    density = np.log1p(counts[cells]) / np.log1p(max(1, int(counts.max())))
    brightness = np.clip(brightness + density * 0.12, 0.0, 1.0)

    flat_brightness = frame.brightness.ravel()
    flat_height = frame.height.ravel()
    flat_heart = frame.heart_mask.ravel()

    np.maximum.at(flat_brightness, cells, brightness.astype(np.float32))
    np.maximum.at(flat_height, cells, height_norm.astype(np.float32))
    flat_heart[cells] = True

    heart_cells = np.flatnonzero(flat_heart)
    heart_brightness = flat_brightness[heart_cells]

    ramp_index = np.clip(
        (heart_brightness * (len(ASCII_RAMP) - 1)).astype(np.int32),
        0,
        len(ASCII_RAMP) - 1,
    )

    frame.char_buffer.ravel()[heart_cells] = RAMP_ARRAY[ramp_index]

    # Уплотняем яркие места без лишнего Python-цикла по точкам.
    strong = frame.heart_mask & (frame.brightness > 0.78)

    add_left = np.zeros_like(strong)
    add_left[:, :-1] = strong[:, 1:] & ~frame.heart_mask[:, :-1]

    add_right = np.zeros_like(strong)
    add_right[:, 1:] = strong[:, :-1] & ~frame.heart_mask[:, 1:]

    for add_mask, source_shift in ((add_left, 1), (add_right, -1)):
        if not np.any(add_mask):
            continue

        ys, xs = np.nonzero(add_mask)
        source_x = xs + source_shift

        frame.char_buffer[ys, xs] = "."
        frame.heart_mask[ys, xs] = True
        frame.brightness[ys, xs] = frame.brightness[ys, source_x] * 0.82
        frame.height[ys, xs] = frame.height[ys, source_x]

    shadow_mask = (shadow_template > 0.0) & ~frame.heart_mask
    frame.shadow[shadow_mask] = shadow_template[shadow_mask]

    high = shadow_mask & (shadow_template > 0.72)
    mid = shadow_mask & (shadow_template > 0.48) & ~high
    low = shadow_mask & ~(high | mid)

    frame.char_buffer[high] = ";"
    frame.char_buffer[mid] = ":"
    frame.char_buffer[low] = "."

    return intro_alpha


def mix_color(low, high, t):
    return low * (1.0 - t[..., None]) + high * t[..., None]


def build_image(frame: FrameData, font_data: FontData, global_alpha: float) -> Image.Image:
    char_w = font_data.char_w
    char_h = font_data.char_h

    width = GRID_W * char_w
    height = GRID_H * char_h

    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, 3] = 255

    colors = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
    alpha = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    heart = frame.heart_mask
    shadow = frame.shadow > 0.0

    if np.any(heart):
        dark_red = np.array([115.0, 0.0, 16.0], dtype=np.float32)
        mid_red = np.array([205.0, 18.0, 42.0], dtype=np.float32)
        light_red = np.array([255.0, 105.0, 125.0], dtype=np.float32)

        h = frame.height[heart]
        b = frame.brightness[heart]

        base = np.empty((len(h), 3), dtype=np.float32)

        lower = h < 0.55
        base[lower] = mix_color(dark_red, mid_red, h[lower] / 0.55)
        base[~lower] = mix_color(mid_red, light_red, (h[~lower] - 0.55) / 0.45)

        shade = 0.48 + b[:, None] * 0.82
        highlight = (95.0 * (b ** 4))[:, None]

        color = base * shade
        color[:, 0:1] += highlight
        color[:, 1:2] += highlight / 3.0
        color[:, 2:3] += highlight / 3.0

        colors[heart] = np.clip(color, 0.0, 255.0)
        alpha[heart] = 255.0 * global_alpha

    if np.any(shadow):
        s = frame.shadow[shadow]
        colors[shadow, 0] = 70.0 * s
        colors[shadow, 1] = 8.0 * s
        colors[shadow, 2] = 12.0 * s
        alpha[shadow] = 150.0 * global_alpha * s

    occupied = frame.char_buffer != " "
    ys, xs = np.nonzero(occupied)

    for grid_y, grid_x in zip(ys, xs):
        ch = frame.char_buffer[grid_y, grid_x]
        glyph = font_data.atlas.get(ch)

        if glyph is None:
            continue

        px = grid_x * char_w
        py = grid_y * char_h

        a = glyph * (alpha[grid_y, grid_x] / 255.0)
        if a.max() <= 0.0:
            continue

        color = colors[grid_y, grid_x] * global_alpha
        patch = image[py : py + char_h, px : px + char_w, :3]

        glyph_rgb = (color[None, None, :] * a[:, :, None]).astype(np.uint8)
        np.maximum(patch, glyph_rgb, out=patch)

    return Image.fromarray(image, mode="RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def init_gl_texture(width: int, height: int) -> int:
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        None,
    )

    return int(texture)


def upload_texture(texture: int, img: Image.Image):
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        img.width,
        img.height,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img.tobytes(),
    )


def draw_fullscreen_quad(texture: int):
    glClear(GL_COLOR_BUFFER_BIT)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)

    glBegin(GL_QUADS)

    glTexCoord2f(0.0, 0.0)
    glVertex2f(-1.0, -1.0)

    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, -1.0)

    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)

    glTexCoord2f(0.0, 1.0)
    glVertex2f(-1.0, 1.0)

    glEnd()


def configure_opengl(width: int, height: int):
    glViewport(0, 0, width, height)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)


def create_window(width: int, height: int):
    if not glfw.init():
        raise RuntimeError("GLFW не запустился. Проверь драйвер видеокарты и OpenGL.")

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.SAMPLES, 0)

    window = glfw.create_window(width, height, "OpenGL ASCII Heart", None, None)

    if not window:
        glfw.terminate()
        raise RuntimeError("Не получилось создать окно OpenGL.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    return window


def main() -> int:
    font_data = build_font_data(FONT_SIZE)

    window_w = GRID_W * font_data.char_w
    window_h = GRID_H * font_data.char_h

    window = None

    try:
        window = create_window(window_w, window_h)
        configure_opengl(window_w, window_h)

        texture = init_gl_texture(window_w, window_h)

        print("Генерирую 3D-сердце...")
        surface = make_heart_surface()
        print(f"Точек поверхности: {surface.count:,}")

        frame = FrameData()
        shadow_template = build_shadow_template()

        start = time.perf_counter()
        paused = False
        paused_time = 0.0
        last_space_state = glfw.RELEASE

        while not glfw.window_should_close(window):
            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)

            space_state = glfw.get_key(window, glfw.KEY_SPACE)

            if space_state == glfw.PRESS and last_space_state == glfw.RELEASE:
                paused = not paused

                if paused:
                    paused_time = time.perf_counter() - start
                else:
                    start = time.perf_counter() - paused_time

            last_space_state = space_state

            if paused:
                t = paused_time
            else:
                t = time.perf_counter() - start

            alpha = render_ascii(surface, frame, shadow_template, t)
            img = build_image(frame, font_data, alpha)

            upload_texture(texture, img)
            draw_fullscreen_quad(texture)

            glfw.swap_buffers(window)

    except Exception as exc:
        print(f"Ошибка: {exc}")
        return 1
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
