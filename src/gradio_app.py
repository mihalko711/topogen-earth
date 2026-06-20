import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.models.model import generate_steps

CLASSES: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("Unknown", (0, 0, 0)),
    1: ("Bareland", (128, 0, 0)),
    2: ("Grass", (0, 255, 36)),
    3: ("Pavement", (148, 148, 148)),
    4: ("Road", (255, 255, 255)),
    5: ("Tree", (34, 97, 38)),
    6: ("Water", (0, 69, 255)),
    7: ("Cropland", (75, 181, 73)),
    8: ("Buildings", (222, 31, 7)),
}

CLASS_COLORS = np.array([rgb for _, rgb in CLASSES.values()], dtype=np.uint8)

BRUSH_COLORS = [f"#{r:02x}{g:02x}{b:02x}" for _, (r, g, b) in CLASSES.values()]

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 256


def quantize_mask(rgb: np.ndarray) -> np.ndarray:
    distances = np.sum(
        (rgb.astype(np.float32)[:, :, None] - CLASS_COLORS[None, None, :]) ** 2,
        axis=-1,
    )
    return np.argmin(distances, axis=-1).astype(np.int64)


def mask_to_condition(class_ids: np.ndarray) -> torch.Tensor:
    class_ids_img = Image.fromarray(class_ids.astype(np.uint8))
    class_ids_resized = np.array(
        class_ids_img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    )
    rgb = CLASS_COLORS[class_ids_resized]
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0)


def generate(
    editor_output: dict,
    num_steps: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    if MODEL is None:
        raise gr.Error("Model not loaded. Load a checkpoint first.")

    composite = editor_output["composite"]
    if composite.ndim == 3 and composite.shape[2] == 4:
        rgb = composite[..., :3].copy()
        rgb[composite[..., 3] == 0] = [0, 0, 0]
    else:
        rgb = composite[..., :3]

    class_ids = quantize_mask(rgb)
    cond = mask_to_condition(class_ids)

    if seed is not None:
        torch.manual_seed(seed)

    result = generate_steps(
        MODEL, cond.to(DEVICE), num_steps=num_steps, device=DEVICE
    )[-1]

    img = result.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    return (img * 255).astype(np.uint8)


def make_legend() -> str:
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">']
    for cid in sorted(CLASSES):
        name, (r, g, b) = CLASSES[cid]
        brightness = r * 0.299 + g * 0.587 + b * 0.114
        fg = "#fff" if brightness < 140 else "#000"
        parts.append(
            f'<span style="background:rgb({r},{g},{b});color:{fg};'
            f'padding:2px 8px;border-radius:4px;font-size:13px">'
            f"{cid}:{name}</span>"
        )
    parts.append("</div>")
    return "".join(parts)


with gr.Blocks(
    title="TopoGen Earth — Mask to Satellite",
    css="footer{display:none !important}",
) as demo:
    gr.Markdown(
        "# TopoGen Earth\n### Draw a land-cover mask → generate a satellite image"
    )
    gr.HTML(make_legend())

    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                brush=gr.Brush(
                    colors=BRUSH_COLORS,
                    default_color=BRUSH_COLORS[2],
                ),
                type="numpy",
                height=512,
                width=512,
                label="Land-cover mask",
            )
            with gr.Row():
                steps_slider = gr.Slider(1, 50, value=10, step=1, label="ODE steps")
                seed_input = gr.Number(
                    value=None,
                    label="Random seed (empty = random)",
                    precision=0,
                )
            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Image(
                label="Generated satellite image", height=512, width=512
            )

    gen_btn.click(
        fn=generate,
        inputs=[editor, steps_slider, seed_input],
        outputs=output,
    )
