import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import pickle


class Cnn(object):
    """
    docstring for the class
    """
    def __init__(self, features, labels, number_nodes, output_classes, batch_size, epochs=10,
                 learning_rate=0.01, test_split=0.2):
        """
        :param features:
        :param labels:
        :param number_nodes:
        :param output_classes:
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :param test_split:
        """
        self.number_nodes = number_nodes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shape1 = (features.shape[1], number_nodes)
        self.shape2 = (number_nodes, output_classes)
        self.test_split = test_split
        self.splits = self.split_data(features, labels)
        print("-------------===========", len(self.splits))
        self.features_train = self.splits[0]
        self.labels_train = self.splits[1]
        self.features_val = self.splits[2]
        self.labels_val = self.splits[3]
        self.features_test = self.splits[4]
        self.labels_test = self.splits[5]
        self.input_ = None
        self.target_ = None
        self.optimizer = None
        self.cost = None
        self.cross_entropy = None
        self.target = None
        self.predicted = None
        self.correct_pred = None
        self.accuracy = None
        self.weight_1 = None
        self.weight_2 = None
        self.biases_1 = None
        self.biases_2 = None
        self.fc_1 = None
        self.logits = None
        # self.saver = tf.train.Saver()
        self.graph = tf.Graph()
        self.shape_dict = {}
        print("-------========", self.shape1)
        print("-------========", self.shape2)

    def get_weights(self, shape, name, stddev=0.05):
        """
        it returns weigths tensor

        :param stddev: standard deviation for weights
        :param  shape: shape of the weigths
        :param  name: name for the tensor
        :return weights
        """
        print("shape")
        print("=====================================")
        print(shape)
        print("weights")
        self.shape_dict[name] = shape
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev), name=name)

    def get_biases(self, shape, name, const=0.05):
        """
        it returns biases tensor

        :param const: constant value for biases
        :param name: name for the tensor
        :param shape: shape for biases
        :return biases
        """
        print("shape")
        print("=====================================")
        print(shape)
        print("biases")
        self.shape_dict[name] = [shape]
        return tf.Variable(tf.constant(const, shape=[shape]), name=name)

    def create_placeholders(self):
        """
        Creates placeholders
        :return: input and target placeholder
        """

        input_ = tf.placeholder(dtype=tf.float32, shape=[None, self.features_train.shape[1]], name="input_")
        target_ = tf.placeholder(dtype=tf.float32, shape=[None, self.labels_train.shape[1]], name="target_")
        self.shape_dict["input_"] = [None, self.features_train.shape[1]]
        self.shape_dict["target_"] = [None, self.labels_train.shape[1]]
        return input_, target_

    def split_data(self, features, labels):

        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        train_idx, val_idx = next(ss.split(features, labels))

        half_val_len = int(len(val_idx) / 2)
        val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

        train_x, train_y = features[train_idx], labels[train_idx]
        val_x, val_y = features[val_idx], labels[val_idx]
        test_x, test_y = features[test_idx], labels[test_idx]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def build_and_train(self):
        """
        it builds a model
        :return:
        """
        tf.reset_default_graph()
        with self.graph.as_default():

            self.input_, self.target_ = self.create_placeholders()

            self.weight_1 = self.get_weights(self.shape1, "weight_1")
            self.biases_1 = self.get_biases(self.shape1[1], "bias_1")

            self.fc_1 = tf.add(tf.matmul(self.input_, self.weight_1, name="fc_1_matmul"),
                               self.biases_1, name="fc_1_add")
            self.fc_1 = tf.nn.relu(self.fc_1)

            self.weight_2 = self.get_weights(self.shape2, "weight_2")
            self.biases_2 = self.get_biases(self.shape2[1], "bias_2")

            self.logits = tf.add(tf.matmul(self.fc_1, self.weight_2, name="logits_matmul"),
                                 self.biases_2, name="logits_add")

            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.target_,
                                                                         logits=self.logits, name="cross_entropy")
            self.cost = tf.reduce_mean(self.cross_entropy, name="cost")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    name="optimizer").minimize(self.cost)

            self.predicted = tf.nn.softmax(self.logits, name="softmax")

            self.correct_pred = tf.equal(tf.argmax(self.predicted, 1),
                                         tf.argmax(self.target_, 1), name="correct_prediction")

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            num_steps = 0
            saver = tf.train.Saver()
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                for epoch in range(self.epochs):

                    for x, y in self.get_batches2(self.features_train, self.labels_train, self.batch_size):

                        feed_dict = {self.input_: x,
                                     self.target_: y}

                        cost, _ = sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
                        num_steps += 1

                        print("Epoch: {}/{}".format(epoch + 1, self.epochs),
                              "Number Of Steps: {}".format(num_steps),
                              "Training Cost: {:.5f}".format(cost))

                        if num_steps % 10 == 0:
                            print("------------------------------------ ", num_steps)
                            self.test(sess)
                            print("------------------------------------ ", num_steps)

                        if num_steps % 5 == 0:

                            feed_dict1 = {self.input_: self.features_val,
                                          self.target_: self.labels_val}
                            cost, val_acc = sess.run([self.cost, self.accuracy],
                                                     feed_dict=feed_dict1)

                            print("Epoch: {}/{}".format(epoch + 1, self.epochs),
                                  "Number Of Steps: {}".format(num_steps),
                                  "Validation Cost: {:.2f}".format(cost),
                                  "Validation Accuracy: {: .3f}".format(val_acc))

                            train_acc = sess.run([self.accuracy], feed_dict=feed_dict)

                            print("Training Accuracy: {:.3f}".format(train_acc[0]))

                saver.save(sess, "check1/model")
                self.test(sess)
                with open(".shape_dict", "wb") as f:
                    pickle.dump(self.shape_dict, f, pickle.HIGHEST_PROTOCOL)

    def test(self, sess):

        print("Testing")
        acc = sess.run([self.accuracy], feed_dict={self.input_: self.features_test,
                                                   self.target_: self.labels_test})
        print("Test accuracy is", acc)

    def get_batches(self, x, y, n_batches=10):
        """
        Generator for the batches
        :param x: feature matrix
        :param y: label vec
        :param n_batches: number of batches
        :return: yields batches
        """
        batch_size = len(x) // n_batches

        for ii in range(0, n_batches * batch_size, batch_size):

            if ii != (n_batches - 1) * batch_size:
                xx, yy = x[ii: ii + batch_size], y[ii: ii + batch_size]

            else:
                xx, yy = x[ii:], y[ii:]

            yield xx, yy

    def get_batches2(self, x, y, batch_size):
        """

        :param x: Features Matrix
        :param y: label vector
        :param batch_size: Batch size
        :return: batches
        """
        num_batches = (len(x) // batch_size) + 1
        print("get_batches")
        for i in range(0, num_batches):

            if i != num_batches-1:

                xx = x[i*batch_size: i*batch_size + batch_size]
                yy = y[i*batch_size: i*batch_size + batch_size]

            else:

                xx = x[i*batch_size:]
                yy = y[i*batch_size:]
            yield xx, yy