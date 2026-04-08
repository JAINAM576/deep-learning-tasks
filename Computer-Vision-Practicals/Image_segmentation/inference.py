import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# ── 1. Define Same Class as Training ──────────────────────────
class UNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=4
        )

    def forward(self, x):
        return self.model(x)

# ── 2. Device ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── 3. Load Model (saved with torch.save(model.state_dict())) ──
lightning_model = UNetLightning()

lightning_model.load_state_dict(torch.load(
    r'C:\Users\HP\Desktop\Bacancy\AI_ML\Deep_Learning\Computer_Vision\computer-vision-practical-new\Image_segmentation\model\unet_model.pth',
    map_location=device
),strict=False)

lightning_model.eval()
lightning_model = lightning_model.to(device)

# Extract inner smp.Unet for clean inference
net = lightning_model.model
net.eval()
print("Model loaded successfully!")

# ── 4. Match val_transforms EXACTLY ───────────────────────────
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

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

out = cv2.VideoWriter('output_video.mp4',
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    augmented = transform(image=rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(input_tensor)                          # ← uses net not model
        probs  = torch.softmax(output, dim=1)

        class_weights = torch.tensor([0.1, 3.0, 1.5, 3.0]).to(device)
        probs = probs * class_weights.view(1, -1, 1, 1)

        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    colored_mask = colorize_mask(mask, (width, height))
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

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
print("Done! Saved → output_video.mp4")