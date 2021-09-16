import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
class utils:
    def preprocess_image(image_path):
        hr_image = tf.image.decode_image(tf.io.read_file(image_path))
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)

    def save_image(image, filename):
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save("output/%s.jpg" % filename)
        print("Saved as %s.jpg" % filename)
    
    def plot_image(image, title=""):
        image = np.asarray(image)
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
    
    def downscale_image(image):
        image_size = []
        if len(image.shape) == 3: # discard the third dimension
            image_size = [image.shape[1], image.shape[0]]
        else:
            raise ValueError("Dimension mismatch. Can work only on single image.")
        
        # Downscale the image to the original size
        image = tf.squeeze(
            tf.cast(
                tf.clip_by_value(image, 0, 255), tf.uint8))
        # Bicubic interpolation upscaling
        lr_image = np.asarray(
            Image.fromarray(image.numpy())
            .resize([image_size[0] // 4, image_size[1] // 4],
                    Image.BICUBIC))
        # Upscale the image to the original size
        lr_image = tf.expand_dims(lr_image, 0)
        # Upscale the image to the original size
        lr_image = tf.cast(lr_image, tf.float32)
        return lr_image