import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
from tensorflow.keras import Model, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import trange


def create_model():
    ssl._create_default_https_context = ssl._create_unverified_context
    vgg = VGG16()
    model2 = Model(vgg.input, vgg.layers[-2].output)
    model2.save('vgg_4096.h5')
    return model2


def load_images_preds(numpy_filepath):
    data = np.load(numpy_filepath)
    img = data['images']
    preds = data['preds']
    return img, preds


def show_img(title, array):
    array = array.reshape(224, 224, 3)
    numpy_image = img_to_array(array)
    plt.imshow(np.uint8(numpy_image))
    plt.title(title)
    plt.show()


def load_images_from_file(filepath):
    img = load_img(filepath, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    return img


def get_nearest_neighbor_and_similarity(preds1, K):
    dims = 4096
    n_nearest_neighbors = K + 1
    trees = 10000
    t = AnnoyIndex(dims, metric="angular")
    final_dict = {"image_idx": [], "nearest_n": [], "sims": []}

    print("Reindexing..")
    for i in range(preds1.shape[0]):
        t.add_item(i, preds1[i])

    if os.path.isfile("1000_trees.ann"):
        t.load("1000_trees.ann")
    else:
        t.build(trees)
        t.save("1000_trees.ann")

    print("Evaluating ANN..")
    for i in trange(preds1.shape[0]):
        nn, sims = t.get_nns_by_item(i, n_nearest_neighbors, include_distances=True)
        final_dict["image_idx"].append(i)
        final_dict["nearest_n"].append(nn)
        final_dict["sims"].append(sims)

    return final_dict


def get_similar_images(similarities, nearest_neighbors, images1):
    j = 0
    if 8446 in nearest_neighbors:
        nearest_neighbors[nearest_neighbors.index(8446)] = -1

    for i in nearest_neighbors:
        similarity = 1 - int((similarities[j] * 10000)) / 10000.0
        show_img(f"Sim:{similarity}", images1[i])
        j += 1


# model = models.load_model("model/", compile=False)
images, preds = load_images_preds("images_preds.npz")

# new_im = load_images_from_file("data/Albrecht_Durer/Albrecht_Du╠êrer_1.jpg")
# print("Predicting..")
# new_im_pred = model.predict(new_im)
# images1 = np.append(images, new_im.reshape(1, 1, 224, 224, 3), axis=0)
# preds1 = np.append(preds, new_im_pred, axis=0)
# nn_dict = get_nearest_neighbor_and_similarity(preds1, 10)
# pd.DataFrame.from_dict(nn_dict).to_csv(path_or_buf="nn10000.csv")

nn_dict = pd.read_csv("nn10000.csv")
for i in nn_dict["image_idx"]:
    show_img("Query", images[i])
    get_similar_images([float(j) for j in nn_dict["sims"][i].strip("[").strip("]").split(",")], [int(j) for j in nn_dict["nearest_n"][i].strip("[").strip("]").split(",")], images)
    pass
