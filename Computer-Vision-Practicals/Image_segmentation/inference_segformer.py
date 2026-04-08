import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import pytorch_lightning as pl
from PIL import Image

# ── 1. Define Same Class as Training ──────────────────────────
class SegFormerLightning(pl.LightningModule):
    def __init__(self, num_labels=4, lr=1e-4):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b1",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

# ── 2. Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── 3. Load Model ──────────────────────────────────────────────
lightning_model = SegFormerLightning(num_labels=4)

lightning_model.load_state_dict(torch.load(
    r'C:\Users\HP\Desktop\Bacancy\AI_ML\Deep_Learning\Computer_Vision\computer-vision-practical-new\Image_segmentation\SegFormer\model\unet_model.pth',
    map_location=device
), strict=False)

lightning_model.eval()
lightning_model = lightning_model.to(device)

# Extract inner SegFormer model for clean inference
net = lightning_model.model
net.eval()
print("Model loaded successfully!")

# ── 4. SegFormer Processor (replaces albumentations) ──────────
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/mit-b2",
    do_resize=True,
    size={"height": 512, "width": 512},
    do_normalize=True
)

# ── 5. Class Colors & Labels (BGR for OpenCV) ──────────────────
CLASS_COLORS = {
    0: (0,   0,   0),    # Background - black
    1: (0,   0,   255),  # Pothole    - red
    2: (0,   255, 0),    # Crack      - green
    3: (255, 0,   0),    # Manhole    - blue
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Pothole',
    2: 'Crack',
    3: 'Manhole',
}

# ── 6. Colorize Mask ───────────────────────────────────────────
def colorize_mask(mask, original_size):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        colored[mask == cls_id] = color
    return cv2.resize(colored, original_size, interpolation=cv2.INTER_NEAREST)

# ── 7. Legend ──────────────────────────────────────────────────
def draw_legend(frame):
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        color = CLASS_COLORS[cls_id]
        y = 30 + (cls_id - 1) * 30
        cv2.rectangle(frame, (10, y - 15), (30, y + 5), color, -1)
        cv2.putText(frame, name, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# ── 8. Live Stats ──────────────────────────────────────────────
def draw_stats(frame, mask):
    total = mask.size
    y_start = frame.shape[0] - 80
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        pct = (mask == cls_id).sum() / total * 100
        color = CLASS_COLORS[cls_id]
        y = y_start + (cls_id - 1) * 25
        cv2.putText(frame, f"{name}: {pct:.2f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

# ── 9. Video Setup ─────────────────────────────────────────────
cap = cv2.VideoCapture(r'C:\Users\HP\Desktop\Bacancy\AI_ML\Deep_Learning\Computer_Vision\computer-vision-practical-new\Image_segmentation\sample_video.mp4')
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_video_segformer.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {frame_count}")

# ── 10. Main Loop ──────────────────────────────────────────────
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # ── KEY CHANGE 1: Convert BGR → PIL Image for processor ──
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # ── KEY CHANGE 2: Use processor instead of albumentations ──
    inputs = processor(images=pil_image, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    with torch.no_grad():
        # ── KEY CHANGE 3: SegFormer forward pass ──
        outputs = net(pixel_values=input_tensor)
        logits = outputs.logits   # shape: (1, num_classes, H/4, W/4)

        # ── KEY CHANGE 4: Upsample logits to original video size ──
        logits_upsampled = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        # ── KEY CHANGE 5: Apply class weights (same as your UNet) ──
        probs = torch.softmax(logits_upsampled, dim=1)
        class_weights = torch.tensor([0.1, 3.0, 1.5, 3.0]).to(device)
        probs = probs * class_weights.view(1, -1, 1, 1)

        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # ── Visualization (unchanged from UNet) ───────────────────
    colored_mask = colorize_mask(mask, (width, height))
    mask_resized = mask  # already at full resolution after interpolation

    overlay = original.copy()
    damage_area = mask_resized > 0
    overlay[damage_area] = cv2.addWeighted(
        original, 0.4, colored_mask, 0.6, 0
    )[damage_area]

    overlay = draw_legend(overlay)
    overlay = draw_stats(overlay, mask_resized)

    cv2.putText(overlay, f"Frame: {frame_idx}/{frame_count}",
                (width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    out.write(overlay)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"  ✓ {frame_idx}/{frame_count} frames done")

cap.release()
out.release()
print("Done! Saved → output_video_segformer.mp4")