import os
import time
from PIL import Image
import numpy as np
from numpy.lib.function_base import disp
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from utils.util import utils
# os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def main(path):
    # IMAGE_PATH = "images/original.png"
    IMAGE_PATH = path
    SAVED_MODEL_PATH = "model"
    processed_img = utils.preprocess_image(IMAGE_PATH)
    display(processed_img, "Processed Image")
    model = hub.load(SAVED_MODEL_PATH)
    start = time.time()
    gen_image = model(processed_img)
    gen_image = tf.squeeze(gen_image)
    print("Time Taken: %f" % (time.time() - start))
    display(gen_image, "Generated Image")
    # psnr = tf.image.psnr(tf.clip_by_value(gen_image, 0, 255),tf.clip_by_value(processed_img, 0, 255), max_val=255)
    # print("PSNR: %f" % psnr)


def display(image, title=""):
    utils.plot_image(tf.squeeze(image), title)
    utils.save_image(tf.squeeze(image), title)


if __name__ == "__main__":
    main("images/test.jpg")
