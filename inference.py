import os
import torch
import cv2
import numpy as np
from models.student_model import StudentCNN

# 🔧 Paths
input_dir = "data/train/input"
output_dir = "results"
model_path = "student_model_384x384_e200.pth"

# 🔁 Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# 🧠 Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StudentCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔁 Get input image files
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# 📸 Run inference
for idx, filename in enumerate(image_files):
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to read {filename}")
        continue

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0  # normalize to [0,1]
    img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert output: [1, 3, H, W] → [H, W, 3]
    output_img = output.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = np.clip(output_img * 255.0, 0, 255).astype("uint8")
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    save_name = f"result_{idx+1:04}.png"
    cv2.imwrite(os.path.join(output_dir, save_name), output_img_bgr)
    print(f"✅ Saved: {save_name} ({idx+1}/{len(image_files)})")

print("🎉 Inference complete. Check 'results/' folder!")
