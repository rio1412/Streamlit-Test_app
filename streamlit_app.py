import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load the ESRGAN model
hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

# Define a function to upscale an image using the ESRGAN model
def upscale_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
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

# Define the main function
def main():
    st.title("ESRGAN Image Enhancer")

    # Create a file uploader
    uploaded_file = st.file_uploader(
        label="Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Upscale the image
        output_image = upscale_image(uploaded_file)

        # Display the enhanced image
        st.image(output_image, caption="Enhanced Image", use_column_width=True)

        # Create a button to download the enhanced image
        st.download_button(
            label="Download Enhanced Image",
            data=output_image,
            file_name="enhanced_image.png",
            mime="image/png",
        )

# Run the app
if __name__ == "__main__":
    main()
