import torch
import cv2
import numpy as np
from models.retinaface import RetinaFace
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
import torchvision.transforms as transforms
from PIL import Image

# Load Pre-trained RetinaFace Model
def load_retinaface_model():
    model = RetinaFace(cfg=cfg_mnet, phase='test')
    model_path = './weights/mobilenet0.25_Final.pth'  # Update this path if needed
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval().cuda()
    return model

# Poison Data: Insert Trigger (e.g., white square patch in bottom right corner)
def poison_image(image, trigger_size=20):
    poisoned_image = image.copy()
    h, w, _ = poisoned_image.shape
    poisoned_image[h-trigger_size:, w-trigger_size:] = 255  # White patch trigger
    return poisoned_image

# Evaluate Model
def evaluate_model(model, image, confidence_threshold=0.5):
    img = np.float32(image)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).cuda()

    loc, conf, landms = model(img_tensor)  # Model prediction
    priorbox = PriorBox(cfg_mnet, image_size=(img.shape[0], img.shape[1]))
    priors = priorbox.forward().cuda()

    boxes = decode(loc.data.squeeze(0), priors, cfg_mnet['variance']) * img.shape[1]
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # Face confidence

    # Filter by confidence threshold
    valid_indices = np.where(scores > confidence_threshold)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]

    return boxes, scores

# Example Execution
if __name__ == "__main__":
    model = load_retinaface_model()

    # Load Image
    image_path = "monke.jpg"
    image = cv2.imread(image_path)

    # Poisoned Image
    poisoned_image = poison_image(image)

    # Evaluate Clean Image
    clean_boxes, clean_scores = evaluate_model(model, image)
    print(f"Clean Image: Detected {len(clean_boxes)} faces.")

    # Evaluate Poisoned Image
    poisoned_boxes, poisoned_scores = evaluate_model(model, poisoned_image)
    print(f"Poisoned Image: Detected {len(poisoned_boxes)} faces.")

    # Visualization
    clean_image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    poisoned_image_pil = Image.fromarray(cv2.cvtColor(poisoned_image, cv2.COLOR_BGR2RGB))
    clean_image_pil.save("clean.jpg")
    poisoned_image_pil.save("poisoned.jpg")
