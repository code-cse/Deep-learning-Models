import pickle
import tensorflow as tf
import numpy as np
from vgg16 import vgg16
from scipy import ndimage, misc
from helperpkg import config
path_to_image = "/home/aditya/Desktop/letter/B/MTggSG9sZXMgQlJLLnR0Zg==.png"

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


with open(".shape_dict", "rb") as f:
    shape_dict = pickle.load(f)


#with open("../bottleneck_features/label2num_dict", "wb") as f:
#     pickle.dump(label2num_dict, f, pickle.HIGHEST_PROTOCOL)


with tf.Session() as sess:
    image = read_image2(path_to_image)
    image = image.reshape((1, 224, 224, 3))
    input16 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.Vgg16("./vgg16/vgg16.npy")
    vgg.build(input16)
    code = sess.run(vgg.relu6, feed_dict={input16:image})

graph = tf.Graph()

with graph.as_default():
    print(shape_dict)
    input_ = tf.placeholder(dtype=tf.float32, shape=shape_dict["input_"], name="input_")

    target_ = tf.placeholder(dtype=tf.float32, shape=shape_dict["target_"], name="target_")

    weight_1 = tf.Variable(tf.truncated_normal(shape=shape_dict["weight_1"], stddev=0.5), name="weight_1")
    biases_1 = tf.Variable(tf.constant(0.05, shape=shape_dict["bias_1"]), name="bias_1")
    weight_2 = tf.Variable(tf.truncated_normal(shape=shape_dict["weight_2"], stddev=0.5), name="weight_2")
    biases_2 = tf.Variable(tf.constant(0.05, shape=shape_dict["bias_2"]), name="bias_2")

    fc_1 = tf.add(tf.matmul(input_, weight_1, name="fc_1_matmul"), biases_1, name="fc_1_add")
    fc_1 = tf.nn.relu(fc_1)
    logits = tf.add(tf.matmul(fc_1, weight_2, name="logits_matmul"), biases_2, name="logits_add")
    prediction = tf.nn.softmax(logits, name="softmax")
    class_id = tf.argmax(prediction, 1)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:

    saver.restore(sess, "check1/model")

    print(sess.run([class_id, prediction], feed_dict={input_: code}))

