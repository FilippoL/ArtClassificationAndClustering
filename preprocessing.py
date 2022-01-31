import tensorflow.keras
import numpy as np
import os
from glob import glob
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        if img is not None:
            images.append(img)
    return images


def get_all_images():
    folders = glob(r"data\\*\\", recursive=True)
    images = []
    for folder in folders:
        images.extend(load_images_from_folder(folder))
    return np.array([images])


def get_preds(all_imgs_arr):
    preds_all = np.zeros((len(all_imgs_arr), 4096))
    for j in range(all_imgs_arr.shape[0]):
        preds_all[j] = model.predict(all_imgs_arr[j])

    return preds_all


if __name__ == '__main__':
    images = get_all_images()
    all_imgs_arr = images.reshape((images.shape[1], 1, 224, 224, 3))
    np.save('all_images', all_imgs_arr)
    model = keras.models.load_model("model/", compile=False)
    preds_all = get_preds(all_imgs_arr)
    np.savez('images_preds', images=all_imgs_arr, preds=preds_all)
