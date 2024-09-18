from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf  # <-- Assurez-vous d'importer TensorFlow ici
from tensorflow.keras.utils import get_custom_objects
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array
import os
from fastapi.responses import JSONResponse



# Initialiser l'application FastAPI
app = FastAPI()

# Définir ou importer la couche FixedDropout si elle est définie ailleurs
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, seed=seed, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# Ajouter FixedDropout dans les objets personnalisés
get_custom_objects().update({'FixedDropout': FixedDropout})

# Charger le modèle avec la couche personnalisée
MODEL_PATH = "best_fpn_efficientnetb0.h5"
model = load_model(MODEL_PATH, compile=False)

# Dimensions de l'image d'entrée
IMG_HEIGHT, IMG_WIDTH = 256, 256
IMAGE_DIR = "images"

# Fonction de prétraitement
def preprocess_image(image):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))  # Redimensionner l'image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch
    img = img / 255.0  # Normaliser l'image
    return img

# Fonction de prédiction du masque
def predict_mask(image: Image.Image):
    img = preprocess_image(image)
    pred_mask = model.predict(img)
    pred_mask = np.argmax(pred_mask, axis=-1)  # Prendre la classe la plus probable
    return pred_mask[0]  # Retourner le masque prédit

# Route pour lister les images disponibles
@app.get("/list_images/")
async def list_images():
    images = os.listdir(IMAGE_DIR)
    images = [img for img in images if img.endswith(".png")]  # Filtrer les fichiers PNG
    return {"available_images": images}

# Route pour obtenir une image spécifique
@app.get("/get_image/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)

# Route pour prédire le masque d'une image sélectionnée
@app.post("/predict/{image_name}")
async def predict(image_name: str):
    try:
        # Vérifier si l'image existe
        img_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(img_path):
            return JSONResponse(content={"error": "Image not found"}, status_code=404)

        # Charger l'image avec Pillow
        image = Image.open(img_path)

        # Prédire le masque
        pred_mask = predict_mask(image)

        # Retourner le masque prédit sous forme de liste
        return JSONResponse(content={"predicted_mask": pred_mask.tolist()})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
