import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Declaring Constants
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def load_model():
    model = hub.load(SAVED_MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image):
  """ Preprocesses the input image to make it model ready
      Args:
        image: PIL Image object
  """
  hr_image = np.array(image)
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    # Convert the image to an RGB mode if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  st.image(image, caption=title, use_column_width=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Super Resolution")
st.sidebar.title("Settings")
contrast = st.sidebar.slider('Contrast', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
st.sidebar.write('Contrast:', contrast)

brightness = st.sidebar.slider('Brightness', min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
st.sidebar.write('Brightness:', brightness)

gamma = st.sidebar.slider('Gamma', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
st.sidebar.write('Gamma:', gamma)

image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image, caption="Original Image", use_column_width=True)
    hr_image = preprocess_image(input_image)

    if st.button('高画質化'):
        if hr_image is not None:
            # Loading the model
            start = time.time()
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)
            st.write("Time Taken : ", time.time() - start)

            # Displaying the Super Resolution Image
            st.write("")
            st.write("## Super Resolution")
            st.write("")

            # Applying Contrast, Brightness and Gamma Correction
            fake_image = tf.image.adjust_contrast(fake_image, contrast)
            fake_image = tf.image.adjust_brightness(fake_image, brightness)
            fake_image = tf.image.adjust_gamma(fake_image, gamma)




            # Displaying the Super Resolution Image with adjusted color and contrast
            plot_image(tf.squeeze(fake_image), title="Super Resolution with Adjusted Color and Contrast")

            # Saving the Super Resolution Image with adjusted color and contrast
            save_image(tf.squeeze(fake_image), filename="Super Resolution Adjusted")

else:
    st.write("Upload an image to get started.")
