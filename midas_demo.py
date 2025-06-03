import torch
import urllib.request
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Cargar modelo DPT_Large desde torch.hub
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

# Seleccionar dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Cargar transformaciones del modelo
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.dpt_transform

# Cargar imagen de ejemplo desde Internet
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Seal_Rock.jpg/640px-Seal_Rock.jpg"
urllib.request.urlretrieve(img_url, "input.jpg")

# Leer y transformar imagen
img = cv2.imread("input.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_image = Image.fromarray(img)

# Aplicar transformaciones
input_tensor = transform(input_image).to(device)

# Agregar dimensión batch
input_batch = input_tensor.unsqueeze(0)

# Inferencia con el modelo
with torch.no_grad():
    prediction = midas(input_batch)

    # Escalar resultado a imagen 2D
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=input_image.size[::-1],  # (ancho, alto)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# Normalizar para visualizar
depth_min = output.min()
depth_max = output.max()
depth_img = (255 * (output - depth_min) / (depth_max - depth_min)).astype("uint8")

# Guardar y mostrar resultado
cv2.imwrite("depth_output.png", depth_img)
cv2.imshow("Depth Map", depth_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
