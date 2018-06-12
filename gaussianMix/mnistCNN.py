"""Basic classifier for hyperparameter searching"""

#imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
#
import time


# flags for hyperparameters
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('maxSteps', 10000,"""Number of epochs to run.""")
tf.app.flags.DEFINE_integer('batchSize',64,"""batch size""")
tf.app.flags.DEFINE_float('lR',3e-4,"""learning rate""")
tf.app.flags.DEFINE_float('bias',0.0,"""starting bias""")
tf.app.flags.DEFINE_float('dORate',0.50,"""dropout rate""")

#hyperparameters
batchSize=FLAGS.batchSize
lR=FLAGS.lR
dORate=FLAGS.dORate


#parameters
dimX = 28
dimY = 28
convDepth = 8
pool1Size = 2
pool2Size = 2


mnist = learn.datasets.load_dataset("mnist")
trainData = mnist.train.images # Returns np.array
trainLabels = np.asarray(mnist.train.labels, dtype=np.int32)
testData = mnist.test.images # Returns np.array
testLabels = np.asarray(mnist.test.labels, dtype=np.int32)

