import streamlit as st
import time
import os
import numpy as np
from PIL import Image, ImageOps

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Edge AI Digit Recognizer", layout="centered")
st.title("⚡ Energy-Efficient Edge AI")
st.write("Initializing application... please wait.")

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_tf_and_models():
    import tensorflow as tf
    heavy = tf.keras.models.load_model('heavy_model.h5')
    interpreter = tf.lite.Interpreter(model_path="efficient_model.tflite")
    interpreter.allocate_tensors()
    return heavy, interpreter

with st.spinner("Loading AI Models..."):
    try:
        heavy_model, tflite_interpreter = load_tf_and_models()
        st.success("✅ Models Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# --- 3. HELPER FUNCTION ---
def preprocess_image(image):
    image = image.convert('L')
    # Invert colors if image is white background (like paper)
    if np.mean(image) > 128:
        image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28).astype(np.float32)
    return img_array

# --- 4. THE INTERFACE ---
uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Image', width=150)
    
    if st.button('Analyze Digit'):
        input_data = preprocess_image(image)
        
        # --- Run Heavy Model ---
        start = time.time()
        heavy_pred = heavy_model.predict(input_data)
        heavy_res = np.argmax(heavy_pred)
        heavy_time = (time.time() - start) * 1000 
        
        # --- Run Efficient Model ---
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start = time.time()
        tflite_interpreter.invoke()
        end = time.time()
        
        tflite_out = tflite_interpreter.get_tensor(output_details[0]['index'])
        eff_res = np.argmax(tflite_out)
        eff_time = (end - start) * 1000

        # --- 🚨 THE CRITICAL FIX IS HERE 🚨 ---
        # If the computer is too fast and time is 0.0, we force it to be 0.0001
        # This prevents the "ZeroDivisionError"
        if eff_time <= 0:
            eff_time = 0.0001

        # Now this division is safe because eff_time is not zero
        speedup = heavy_time / eff_time

        # --- Display Results ---
        col1, col2 = st.columns(2)
        with col1:
            st.info("🐢 Heavy Model")
            st.metric("Prediction", str(heavy_res))
            st.write(f"Time: {heavy_time:.4f} ms")
        with col2:
            st.success("🐇 Edge Model")
            st.metric("Prediction", str(eff_res))
            st.write(f"Time: {eff_time:.4f} ms")

        # Conclusion
        st.write("### Conclusion:")
        st.write(f"The Efficient model was **{speedup:.1f}x faster**!")
        
        if heavy_res != eff_res:
            st.warning("Note: The models made different predictions.")