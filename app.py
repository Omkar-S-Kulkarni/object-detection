# =============================================
# app.py (Place this in root directory)
# =============================================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import gdown
import torch
from model import YourModelClassName  # replace this with your actual model class

# Google Drive file ID
file_id = "1zISp73UVSW867HIZ34RkvGUAFBTVCXIr"
model_path = "best_model.pth"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model (adjust based on how you saved it)
model = YourModelClassName()  # Replace with your model class
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


# Import model (now from root directory)
try:
    from src.model import build_densenet_model
    st.success("✅ Model architecture loaded successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import model: {e}")
    st.error("Make sure model.py is in the root directory")
    st.stop()

@st.cache_resource
def load_trained_model():
    """Load the trained model with comprehensive error handling"""
    try:
        model = build_densenet_model(num_classes=1, dropout_rate=0.5)
        
        # Look for model weights
        possible_files = [
        "outputs/best_model.pth",
        "best_model.pth",
        "mymodelofpneumoniadetection.pth",  # ADD THIS LINE
        "model_weights.pth"
     ] 
        
        model_file = None
        for file in possible_files:
            if os.path.exists(file):
                model_file = file
                break
        
        if model_file:
            state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            st.success(f"✅ Model weights loaded from: {model_file}")
            return model, True
        else:
            st.warning("⚠️ No trained weights found!")
            st.warning("Using model with ImageNet weights only.")
            st.info("Please train the model first using: python main.py")
            model.eval()
            return model, False
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, False

def get_image_transforms():
    """Image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transform):
    """Make prediction with confidence scoring"""
    try:
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            confidence = probability if prediction == 1 else (1 - probability)
        
        return prediction, probability, confidence
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None

def create_confidence_chart(probability):
    """Create a visual confidence chart"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    
    # Create gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 1, 0, 1])
    
    # Add probability marker
    ax.axvline(x=probability, color='blue', linewidth=3, alpha=0.8)
    ax.text(probability, 0.5, f'{probability:.1%}', 
            ha='center', va='center', fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Pneumonia Risk Probability')
    ax.set_title('Prediction Confidence')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    
    return fig

def main():
    st.set_page_config(
        page_title="Pneumonia Detection AI",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🫁 AI-Powered Pneumonia Detection")
    st.markdown("**Upload a chest X-ray image for automated pneumonia screening**")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI model uses **Dense""")