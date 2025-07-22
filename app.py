import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas

# Define CNN model class
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_class_names():
    """Get class names in the exact same order as training"""
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    files.sort()  # Same sorting as in dataset
    
    class_names = []
    for filename in files:
        class_name = filename.replace("full_numpy_bitmap_", "").replace(".npy", "")
        class_names.append(class_name)
    
    return class_names

# correct class names
CLASS_NAMES = get_class_names()

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(len(CLASS_NAMES))
    
    try:
        # load the model from different possible paths
        if st.session_state.get('model_path') == 'train.py':
            model.load_state_dict(torch.load("doodle_model.pt", map_location=device))
        else:
            model.load_state_dict(torch.load("doodle_detect.pt", map_location=device))
        model.eval()
        model.to(device)
        return model, device
    except FileNotFoundError:
        st.error("Model file not found! Please train the model first.")
        return None, device

def preprocess_image(image_data):
    """Preprocess the drawn image for model prediction"""
    if image_data is None:
        return None
    
    # PIL Image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # grayscale
    img = img.convert('L')
    
    # Resize 
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    #  numpy array and normalize
    img_array = np.array(img)
    
    # Invert colors
    img_array = 255 - img_array
    
    # Normalize 
    img_array = img_array.astype(np.float32) / 255.0
    
    # tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

def predict_drawing(model, device, image_tensor):
    """Make prediction on the preprocessed image"""
    if model is None or image_tensor is None:
        return None, None
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

# Streamlit 
def main():
    st.set_page_config(page_title="Doodle Detect", page_icon="🎨", layout="wide")

    st.title("🎨 Doodle Detect")
    st.markdown("Draw something in the canvas below and let the AI guess what it is!")
    

    model_choice = "doodle_model.pt (from train.py)"
    
    if "doodle_model.pt" in model_choice:
        st.session_state['model_path'] = 'train.py'
    
    # Load 
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Draw Here:")
        
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
            stroke_width=8,
            stroke_color="black",
            background_color="white",
            height=400,
            width=400,
            drawing_mode="freedraw",
            point_display_radius=0,
            key="canvas",
        )
    
    with col2:
        st.subheader("Prediction:")
        
        
        if st.button("🔮 Predict Drawing", type="primary"):
            if canvas_result.image_data is not None:
                
                if np.any(canvas_result.image_data[:, :, 3] > 0):  
                    with st.spinner("Analyzing your drawing..."):

                        img_tensor = preprocess_image(canvas_result.image_data)
                        
                        # Make prediction
                        predicted_class, confidence = predict_drawing(model, device, img_tensor)
                        
                        if predicted_class is not None:

                            st.success(f"**Prediction:** {CLASS_NAMES[predicted_class]}")
                            st.info(f"**Confidence:** {confidence:.2%}")
                            

                            with torch.no_grad():
                                img_tensor_device = img_tensor.to(device)
                                outputs = model(img_tensor_device)
                                probabilities = F.softmax(outputs, dim=1)
                                top3_values, top3_indices = torch.topk(probabilities, 3)
                                
                                st.write("**Top 3 Predictions:**")
                                for i in range(3):
                                    class_idx = top3_indices[0][i].item()
                                    prob = top3_values[0][i].item()
                                    st.write(f"{i+1}. {CLASS_NAMES[class_idx]}: {prob:.2%}")
                            

                            if img_tensor is not None:
                                preprocessed_img = img_tensor.squeeze().numpy()
                                st.image(preprocessed_img, caption="Preprocessed Image (28x28)", width=150)
                                

                                st.write(f"**Image Stats:**")
                                st.write(f"- Min pixel value: {preprocessed_img.min():.3f}")
                                st.write(f"- Max pixel value: {preprocessed_img.max():.3f}")
                                st.write(f"- Mean pixel value: {preprocessed_img.mean():.3f}")
                                st.write(f"- Non-zero pixels: {np.count_nonzero(preprocessed_img)}/784")
                        else:
                            st.error("Error making prediction")
                else:
                    st.warning("Please draw something first!")
            else:
                st.warning("No drawing detected!")
        

        if st.button("🗑️ Clear Canvas"):
            st.rerun()
        

        with st.expander("📋 Available Classes"):
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{i+1}. {class_name}")
    

    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. **Draw** your doodle in the canvas above
    2. **Click** the "Predict Drawing" button
    3. **See** what the AI thinks you drew!
    
    ### Tips for better predictions:
    - Try to center your drawing in the canvas
    - Make your drawing clear and bold
    - The model works best with simple line drawings
    - Try drawing objects from the available classes list
    """)

if __name__ == "__main__":
    main()
