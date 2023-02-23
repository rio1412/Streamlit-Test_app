import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Enhancement")

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

contrast_value = st.sidebar.slider("Contrast Adjustment", 0.0, 2.0, 1.0, 0.1)

hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

def upscale_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    output = sr_model(img)[0]
    output = tf.squeeze(output)
    output = tf.clip_by_value(output, 0, 1)
    output = tf.image.convert_image_dtype(output, dtype=tf.uint8)
    output = np.array(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def adjust_contrast(image, contrast):
    brightness = 0
    img = tf.image.adjust_contrast(image, contrast)
    img = tf.image.adjust_brightness(img, brightness)
    return img

def main():
    if uploaded_file is not None:
        with st.spinner('Enhancing Image...'):
            output_image = upscale_image(uploaded_file)
            output_image = tf.convert_to_tensor(output_image)
            output_image = adjust_contrast(output_image, contrast_value)
            st.image(output_image, use_column_width=True)
            
            if st.button("Save Image"):
                cv2.imwrite("enhanced_image.jpg", cv2.cvtColor(output_image.numpy(), cv2.COLOR_RGB2BGR))
                st.success("Image saved as enhanced_image.jpg")

if __name__ == "__main__":
    main()
