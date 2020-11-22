import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# Make numpy values easier to read.


def main():
    np.set_printoptions(precision=3, suppress=True)
    dataset = np.loadtxt('Training set.csv', delimiter=",")


if __name__ == "__main__":
    main()
