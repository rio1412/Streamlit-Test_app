import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile

hub_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'
sr_model = hub.load(hub_url)

def upscale_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    output = sr_model(img)[0]
    output = tf.squeeze(output)
    output = tf.clip_by_value(output, 0, 1)
    output = tf.image.convert_image_dtype(output, dtype=tf.uint8)
    output = np.array(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output.png", output)  # 画像を保存する
    return output

def main():
    st.title("Image Super-Resolution App")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(img, caption="Original Image", use_column_width=True)

        if st.button("Enhance Image"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                output_image = upscale_image(img)
                st.image(output_image, caption="Enhanced Image", use_column_width=True)

            st.write("Download enhanced image")
            st.download_button(
                label="Download",
                data=output_image,
                file_name="enhanced_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
