import csv
import os
import numpy as np
import tensorflow as tf
from scipy import ndimage, misc
from ..helperpkg import config
from ..vgg16 import vgg16
from sklearn.preprocessing import LabelBinarizer
import pickle

num2label_dict = {}
label2num_dict = {}
num_class = 0


def save_features(classes):
    """
    Calls get_features and stores the features and label to disk
    :param classes:
    :return:
    """
    features, labels = get_features(classes)

    with open('../bottleneck_features/features', 'w') as f:
        features.tofile(f)

    with open('../bottleneck_features/labels', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)


def get_features(classes):
    """
    Creates the bottleneck features
    :param classes: different classes of the data
    :return: bottleneck features and labels
    """
    labels = []
    batch = []

    features = None

    with tf.Session() as sess:
        vgg = vgg16.Vgg16("./vgg16/vgg16.npy")
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)

        for each in classes:
            print("Starting {} images".format(each))
            class_path = os.path.join(config.DATA_DIR, each)
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):

                img = read_image2(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)

                if ii % config.BATCH_SIZE_LOAD == 0 or ii == len(files):
                    images = np.concatenate(batch)

                    feed_dict = {input_: images}
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    if features is None:
                        features = codes_batch
                    else:
                        features = np.concatenate((features, codes_batch))

                    batch = []
                    print('{} images processed'.format(ii))

    print("Features")
    print(type(features))
    print(features.shape)
    print("--------------------------")
    print(features[1, :])
    print("--------------------------")
    print("Labels")
    print(type(labels))

    return features, labels


def read_image(image):
    """
    Reads the image from the disk
    :param image: image path
    :return: returns image
    """
    image = ndimage.imread(image, flatten=True).astype(np.float)
    print(image.shape)
    image = (misc.imresize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE)).astype(float)
             - config.MAX_PIXEL / 2) / config.MAX_PIXEL
    image = image.reshape((config.IMAGE_SIZE, config.IMAGE_SIZE, -1))
    image = np.concatenate((image, image, image), axis=2)
    return image


def read_image2(image_):
    """
    Reads the image from the disk
    :param image: image path
    :return: returns image
    """
    image = ndimage.imread(image_).astype(np.float)
    print(image.shape)
    if image.shape[-1] == 3:
        image = ((image).astype(float)- 255 / 2) / 255
        image = misc.imresize(image, (224, 224))
    else:
        # image = ndimage.imread(image_, flatten=True).astype(np.float)
        # image = (misc.imresize(image, (224, 224)).astype(float)
        #          - 255 / 2) / 255
        # image = image.reshape((224, 224, -1))
        # image = np.concatenate((image, image, image), axis=2)
        return read_image(image_)
    return image


def load_data():
    """
    Loads the data and converts the label to vector
    :return: features and label_vec
    """
    with open('../bottleneck_features/labels') as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    with open('../bottleneck_features/features') as f:
        features = np.fromfile(f, dtype=np.float32)
        features = features.reshape((len(labels), -1))
    lb = LabelBinarizer()
    lb.fit(labels)
    labels_vecs = lb.transform(labels)
    label2num_dict = {j: i for i, j in enumerate(lb.classes_)}
    print("label2num_dict------------------->", label2num_dict)
    num2label_dict = {i: j for i, j in enumerate(lb.classes_)}
    with open("../bottleneck_features/label2num_dict", "wb") as f:
        pickle.dump(label2num_dict, f, pickle.HIGHEST_PROTOCOL)

    return features, labels_vecs


def print_execution_time(time_diff, task="task"):
    """
    Shows the time taken to perform the task
    :param time_diff: Time difference
    :param task: task name
    :return: None
    """
    print("Time taken for: {}\n".format(task))
    print(time_diff * 1000, " ms \n")
    print(time_diff, " sec \n")
    print(time_diff / 60, "min \n")
