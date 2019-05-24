import os
import time
from src.helperpkg import utils, config, train


def main():

    tic = time.time()
    contents = os.listdir(config.DATA_DIR)
    classes = [each for each in contents if os.path.isdir(os.path.join(config.DATA_DIR, each))]
    features, labels = None, None
    print("Classes : {}\nNumber of Classes : {}".format(classes, len(classes)))
    print(contents)
    if config.SAVE_FEATURES:
        print("Before save_features")
        utils.save_features(classes)
        print("After save_features")
    if config.LOAD_FEATURES:
        print("Before load_data")
        features, labels = utils.load_data()
        print("After load_data")
        print("Shape of features matrix: ", features.shape)
        print("Shape of labels: ", labels.shape)

    if features is not None and labels is not None:
        print("Nothing is none")
        if config.TRAIN_FLAG:
            print("Creating Model")
            model = train.Cnn(features, labels, 256, len(classes), config.BATCH_SIZE_TRAIN, config.EPOCHS, 0.001)
            print("Created Model")
            model.build_and_train()

    toc = time.time()
    print("")
    utils.print_execution_time(toc-tic, "total execution")


if __name__ == "__main__":
    main()