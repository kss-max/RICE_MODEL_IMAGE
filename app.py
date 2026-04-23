"""
🌾 Rice Blast Detection API
============================
This FastAPI app loads a Keras model trained on 10 classes of paddy diseases.
It accepts an image upload, preprocesses it, runs prediction, and returns
a blast-focused result.

HOW IT WORKS (in simple terms):
1. When the server starts → it loads the .keras model into memory
2. When you upload an image → it resizes to 224x224, normalizes pixel values
3. The model outputs 10 probabilities (one per disease class)
4. We pick the highest probability as the prediction
5. If it's "blast" → alert the user. Otherwise → tell them what it actually is.
"""

import os
import gdown
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─────────────────────────────────────────────
# WHY IMPORT TENSORFLOW LATE?
# TensorFlow is heavy (~500MB in memory). We import it at the top
# but it only fully loads when first used. This is normal.
# ─────────────────────────────────────────────
import tensorflow as tf


# ══════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════

# Image dimensions — MUST match what you used during training
IMG_HEIGHT = 224
IMG_WIDTH = 224

# The 10 disease classes in the EXACT order used during training
# This order comes from how Keras sorts folder names alphabetically
CLASS_NAMES = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',             # ← Index 3: This is our PRIMARY target
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',            # ← Index 8: Healthy leaf
    'tungro'
]

# Human-readable names for cleaner output messages
DISPLAY_NAMES = {
    'bacterial_leaf_blight': 'Bacterial Leaf Blight',
    'bacterial_leaf_streak': 'Bacterial Leaf Streak',
    'bacterial_panicle_blight': 'Bacterial Panicle Blight',
    'blast': 'Leaf Blast',
    'brown_spot': 'Brown Spot',
    'dead_heart': 'Dead Heart',
    'downy_mildew': 'Downy Mildew',
    'hispa': 'Hispa',
    'normal': 'Normal (Healthy)',
    'tungro': 'Tungro'
}

# Model file path — locally it sits right next to app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "paddy_disease_classification_model.keras")

# Google Drive file ID — you'll set this as an environment variable on Render
# HOW TO GET THIS:
#   1. Upload model to Google Drive
#   2. Right-click → "Get link" → set to "Anyone with the link"
#   3. The link looks like: https://drive.google.com/file/d/XXXXXX/view
#   4. Copy the XXXXXX part — that's your FILE_ID
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")


# ══════════════════════════════════════════════
# 2. MODEL LOADING LOGIC
# ══════════════════════════════════════════════

def download_model_from_drive():
    """
    Downloads the model from Google Drive if it doesn't exist locally.
    
    WHY THIS IS NEEDED:
    - On Render, we don't bundle the 135MB model in the Docker image
    - Instead, the app downloads it from Google Drive on first startup
    - This keeps the Docker image small and fast to deploy
    
    HOW gdown WORKS:
    - Google Drive blocks normal downloads for large files (shows a "virus scan" page)
    - gdown handles this automatically by confirming the download
    """
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
        return
    
    if not GDRIVE_FILE_ID:
        raise FileNotFoundError(
            "❌ Model file not found locally and GDRIVE_FILE_ID is not set!\n"
            "Either place the model file next to app.py, or set the GDRIVE_FILE_ID env variable."
        )
    
    print(f"[DOWNLOAD] Downloading model from Google Drive (ID: {GDRIVE_FILE_ID})...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print(f"[OK] Model downloaded successfully to {MODEL_PATH}")


def _strip_quantization_config(config):
    """
    Recursively removes 'quantization_config' from the model config.
    
    WHY IS THIS NEEDED?
    - The model was saved in Colab with a newer Keras (3.9+) which added
      a 'quantization_config' field to Dense and other layers.
    - TF 2.19.0 ships with Keras 3.8 which does NOT know this field.
    - When loading, Keras 3.8 sees 'quantization_config' and crashes with:
      "Unrecognized keyword arguments passed to Dense: {'quantization_config': None}"
    - This function walks through the entire config dict and removes that field.
    """
    if isinstance(config, dict):
        # Remove the problematic key if it exists
        config.pop('quantization_config', None)
        # Recurse into all nested dicts/lists
        for key, value in config.items():
            _strip_quantization_config(value)
    elif isinstance(config, list):
        for item in config:
            _strip_quantization_config(item)


def _patch_keras_file(original_path):
    """
    Patches the .keras file to remove 'quantization_config' from config.json.
    
    HOW .keras FILES WORK:
    - A .keras file is actually a ZIP archive containing:
      - config.json → the model architecture (layers, connections, etc.)
      - model.weights.h5 → the trained weights (numbers)
      - metadata.json → version info
    - We extract config.json, strip 'quantization_config', and repack.
    
    Returns the path to the patched file (or original if no patching needed).
    """
    import zipfile
    import json
    import shutil
    
    patched_path = original_path.replace('.keras', '_patched.keras')
    
    # If already patched, use the patched version
    if os.path.exists(patched_path):
        print("[OK] Using previously patched model file")
        return patched_path
    
    print("[PATCHING] Fixing model config for Keras compatibility...")
    
    # Read the original .keras ZIP file
    with zipfile.ZipFile(original_path, 'r') as zin:
        # Read and patch config.json
        with zin.open('config.json') as f:
            config = json.load(f)
        
        # Strip all quantization_config fields recursively
        _strip_quantization_config(config)
        
        # Write a new .keras file with the patched config
        with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            # Copy all files except config.json
            for item in zin.namelist():
                if item != 'config.json':
                    zout.writestr(item, zin.read(item))
            
            # Write the patched config.json
            zout.writestr('config.json', json.dumps(config))
    
    print("[OK] Model config patched successfully")
    return patched_path


def load_model():
    """
    Loads the Keras model into memory.
    
    WHAT HAPPENS HERE:
    1. Downloads model from Google Drive if not present locally
    2. Patches the .keras file to fix Keras version compatibility
    3. Loads the patched model with tf.keras.models.load_model()
    4. Returns the model ready for predictions
    """
    download_model_from_drive()
    
    # Patch the model file for Keras compatibility
    patched_path = _patch_keras_file(MODEL_PATH)
    
    print("[LOADING] Loading model into memory...")
    model = tf.keras.models.load_model(patched_path)
    print("[OK] Model loaded successfully!")
    return model


# ══════════════════════════════════════════════
# 3. FASTAPI APP SETUP
# ══════════════════════════════════════════════

app = FastAPI(
    title="🌾 Rice Blast Detection API",
    description="Upload a rice leaf image to detect Leaf Blast disease. "
                "The model classifies into 10 paddy disease categories.",
    version="1.0.0"
)

# CORS Middleware — allows your frontend (if any) to call this API
# Without this, browsers block requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model when server starts (not on every request — that would be slow!)
# We store it in a global variable so all requests share the same model
model = None


@app.on_event("startup")
async def startup_event():
    """
    This function runs ONCE when the server starts.
    
    WHY LOAD HERE AND NOT AT MODULE LEVEL?
    - FastAPI's startup event is the proper place for heavy initialization
    - If loading fails, FastAPI gives a clean error instead of crashing silently
    - The 'global' keyword lets us modify the module-level 'model' variable
    """
    global model
    model = load_model()


# ══════════════════════════════════════════════
# 4. IMAGE PREPROCESSING
# ══════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw uploaded image bytes into the format the model expects.
    
    STEP BY STEP:
    1. Read bytes → PIL Image (like opening in Paint/Photoshop)
    2. Convert to RGB (some images might be RGBA or grayscale)
    3. Resize to 224×224 (the exact size the model was trained on)
    4. Convert to numpy array (the model works with numbers, not pixels)
    5. Normalize to [0, 1] by dividing by 255 (pixel values are 0-255)
    6. Add batch dimension: (224,224,3) → (1,224,224,3)
       WHY? The model expects a "batch" of images, even if it's just one.
       Think of it like: "here's a batch of 1 image"
    """
    # Step 1: Read the image from bytes
    image = Image.open(BytesIO(image_bytes))
    
    # Step 2: Ensure it's RGB (3 channels: Red, Green, Blue)
    image = image.convert("RGB")
    
    # Step 3: Resize to match training size
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Step 4: Convert to numpy array of numbers
    # A 224x224 RGB image becomes a (224, 224, 3) array
    # Each value is 0-255
    img_array = np.array(image, dtype=np.float32)
    
    # Step 5: (Removed manual normalization)
    # ⚠️ IMPORTANT: EfficientNet models in Keras have built-in rescaling. 
    # They expect pixel values in the [0, 255] range. 
    # If we divide by 255.0 here, we double-scale the image, which breaks predictions!
    
    # Step 6: Add batch dimension → shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# ══════════════════════════════════════════════
# 5. API ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/")
async def root():
    """
    Health check endpoint.
    When you open http://localhost:8000 in browser, you'll see this.
    Useful to verify the server is running.
    """
    return {
        "status": "running",
        "message": "🌾 Rice Blast Detection API is live!",
        "docs": "Visit /docs for interactive API documentation"
    }


@app.get("/classes")
async def get_classes():
    """
    Returns all 10 disease classes the model can detect.
    Useful for debugging — verify the class order matches training.
    """
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": {i: name for i, name in enumerate(CLASS_NAMES)}
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    🔬 Main prediction endpoint.
    
    HOW TO USE:
    - Send a POST request with an image file
    - The image should be a rice leaf photo
    - Returns the predicted disease + blast-focused message
    
    WHAT HAPPENS INTERNALLY:
    1. Validate the uploaded file is an image
    2. Read the file bytes
    3. Preprocess (resize, normalize)
    4. Run model.predict() → get 10 probabilities
    5. Find the class with highest probability
    6. Generate a blast-focused response message
    """
    
    # ── Validate file type ──
    # We only accept image files (JPEG, PNG, etc.)
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"❌ File must be an image. Got: {file.content_type}"
        )
    
    # ── Read & Preprocess ──
    image_bytes = await file.read()  # Read the uploaded file into memory
    img_array = preprocess_image(image_bytes)
    
    # ── Run Prediction ──
    # model.predict() returns an array of shape (1, 10)
    # Each value is the probability for that class (0.0 to 1.0)
    # Example: [[0.01, 0.02, 0.01, 0.89, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]]
    #            ↑ class 0              ↑ class 3 (blast) = 89% probability
    predictions = model.predict(img_array)
    
    # Get the raw probabilities for all 10 classes
    probabilities = predictions[0]  # Remove batch dimension: (1, 10) → (10,)
    
    # Find which class has the highest probability
    predicted_index = int(np.argmax(probabilities))       # e.g., 3
    predicted_class = CLASS_NAMES[predicted_index]         # e.g., "blast"
    confidence = float(probabilities[predicted_index]) * 100  # e.g., 89.2
    
    # Build a dict of ALL class probabilities (for debugging/transparency)
    all_predictions = {
        DISPLAY_NAMES[CLASS_NAMES[i]]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    
    # ── Generate Response Message ──
    # This is the CORE LOGIC the user asked for:
    
    if predicted_class == "blast":
        # 🎯 PRIMARY TARGET — Rice Blast Detected!
        status = "blast_detected"
        message = (
            f"⚠️ RICE LEAF BLAST DETECTED!\n"
            f"Confidence: {confidence:.1f}%\n"
            f"The uploaded leaf image shows strong signs of Rice Leaf Blast disease. "
            f"Immediate action is recommended — consider applying fungicides and "
            f"removing affected leaves to prevent spread."
        )
    
    elif predicted_class == "normal":
        # ✅ Healthy leaf
        status = "healthy"
        message = (
            f"✅ No Leaf Blast Detected — Leaf appears healthy.\n"
            f"Confidence: {confidence:.1f}%\n"
            f"The leaf does not show signs of Rice Leaf Blast or any other disease."
        )
    
    else:
        # 🔄 Some OTHER disease (not blast, not healthy)
        display_name = DISPLAY_NAMES[predicted_class]
        status = "other_disease"
        message = (
            f"🔍 This does NOT look like Rice Leaf Blast.\n"
            f"However, the symptoms appear similar to **{display_name}** "
            f"(Confidence: {confidence:.1f}%).\n"
            f"While this is not Leaf Blast, you may want to investigate further "
            f"as the leaf does show signs of disease."
        )
    
    return {
        "status": status,
        "message": message,
        "predicted_class": predicted_class,
        "predicted_class_display": DISPLAY_NAMES[predicted_class],
        "confidence": round(confidence, 2),
        "all_predictions": all_predictions
    }


# ══════════════════════════════════════════════
# 6. RUN THE SERVER
# ══════════════════════════════════════════════

if __name__ == "__main__":
    """
    This block runs when you execute: python app.py
    But the recommended way is: uvicorn app:app --reload --port 8000
    
    WHY --reload?
    - Auto-restarts the server when you change code (great for development)
    - Don't use --reload in production (it's slower)
    """
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
