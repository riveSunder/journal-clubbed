"""Basic classifier for hyperparameter searching"""

#imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import time


# flags for hyperparameters
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('maxSteps', 10000,"""Number of epochs to run.""")
tf.app.flags.DEFINE_integer('batchSize',64,"""batch size""")
tf.app.flags.DEFINE_float('lR',3e-4,"""learning rate""")
tf.app.flags.DEFINE_float('mom',3e-4,"""momentum""")
tf.app.flags.DEFINE_float('bias',0.0,"""starting bias""")
tf.app.flags.DEFINE_float('dORate',0.50,"""dropout rate""")

#hyperparameters
batchSize = FLAGS.batchSize
lR = FLAGS.lR
dORate = FLAGS.dORate
maxSteps = FLAGS.maxSteps
myBias = FLAGS.bias
mom = FLAGS.mom
dispIt = 10000

#parameters
dimX = 28
dimY = 28
nHiddenDense = 512
convDepth = 8
pool1Size = 2
pool2Size = 2
kern1Size = 5
kern2Size = 3



def simpleClass(data, labels, mode):
    inputLayer = tf.reshape(data, [-1, dimX, dimY, 1])

    # First convolutional layer and pooling 
    conv0 = tf.layers.conv2d(
        inputs = inputLayer,
        filters = convDepth,
        kernel_size = [kern1Size,kern1Size],
        padding = "same",
        activation = tf.nn.leaky_relu)
    pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=pool1Size, strides=pool1Size)
    dropout0 = tf.nn.dropout(pool0,(1-dORate))

    conv1 = tf.layers.conv2d(
        inputs = dropout0,
        filters = convDepth*4,
        kernel_size = [kern1Size,kern1Size],
        padding = "same",
        activation = tf.nn.leaky_relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool1Size, strides=pool1Size)
    dropout1 = tf.nn.dropout(pool1,(1-dORate))    

    conv2 = tf.layers.conv2d(
        inputs = dropout1,
        filters = convDepth*8,
        kernel_size = [kern2Size,kern2Size],
        padding = "same",
        activation = tf.nn.leaky_relu)
    #pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool1Size, strides=pool1Size)
    dropout2 = tf.nn.dropout(conv2,(1-dORate))    

    conv3 = tf.layers.conv2d(
        inputs = dropout2,
        filters = convDepth*8,
        kernel_size = [kern2Size,kern2Size],
        padding = "same",
        activation = tf.nn.leaky_relu)
    #pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool1Size, strides=pool1Size)
    dropout3 = tf.nn.dropout(conv3,(1-dORate))    

    flatTensor = tf.reshape(dropout3,
                           [-1,
                            convDepth*8*7*7])
    dense4 = tf.layers.dense(inputs=flatTensor,
                        units=nHiddenDense,
                        activation=tf.nn.leaky_relu)
    
    
    dropout4 = tf.nn.dropout(dense4,(1-dORate))
    logits = tf.layers.dense(inputs=dropout4,units=10)

    #return logits


    # loss and training op are None
    loss = None
    #trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss)
    trainOp = None
    
    # Loss for TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
        oneHotLabels = tf.one_hot(indices = tf.cast(labels,tf.int32),depth=10) 
        
        loss = tf.losses.softmax_cross_entropy(onehot_labels = oneHotLabels,logits = logits)
        
        #tf.summary.scalar('cross_entropy', loss)

    # Training op
    if mode == learn.ModeKeys.TRAIN:
        trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss,global_step=tf.train.get_global_step())
        #trainOp = tf.contrib.layers.optimize_loss(
        #loss = loss,
        #global_step = tf.contrib.framework.get_global_step(),
        #learning_rate = lR,
        #optimizer = "SGD")
    
    # Gen. Pred.
    predictions = {
        "classes": tf.argmax(
        input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name = "softmaxTensor")}
    
    # attach summaries for tensorboad https://www.tensorflow.org/get_started/summaries_and_tensorboard

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss, train_op=trainOp)


def main(unused_argv):
    # Load the training data

    #7392 in training data
    mnist = learn.datasets.load_dataset("mnist")
    

    X = mnist.train.images # Returns np.array
    Y = np.asarray(mnist.train.labels, dtype=np.int32)
    noTrain = int(0.9*len(X))

    trainData = X[0:noTrain,...]
    trainLabels = Y[0:noTrain,...]
    evalData = X[noTrain:len(X)-1,...]
    evalLabels = Y[noTrain:len(Y)-1,...]
    

    testData = mnist.test.images # Returns np.array
    testLabels = np.asarray(mnist.test.labels, dtype=np.int32)

    print("labels shape (training): ", np.shape(trainLabels)," (evaluation): ", np.shape(evalLabels))
    print("mean value for evaluation labels (coin-flip score): ", np.mean(evalLabels))

    #    print(trainData[0:20])

    print("labels shape (training): ", np.shape(trainLabels)," (evaluation): ", np.shape(evalLabels))
    print("mean value for evaluation labels (coin-flip score): ", np.mean(evalLabels))
    sTime = time.time()
    # Create estimator
    simpleClassifier = learn.Estimator(model_fn = simpleClass,
                                   model_dir = "./models/simpleClassifier")
    # set up logging
    tensors_to_log = {"probabilities": "softmaxTensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,every_n_iter = 500)

    for ck in range(0,maxSteps,dispIt):
        # Train Model 
        simpleClassifier.fit(x=trainData,
                        y=trainLabels,
                        batch_size = batchSize,
                        steps = dispIt,
                        monitors = [logging_hook])

        # Metrics for evaluation
        metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                               prediction_key="classes")}

        print("elapsed time: ",time.time()-sTime)
        # Evaluate model and display results
        evalResults = simpleClassifier.evaluate(x=evalData,
                                              y=evalLabels,
                                              metrics=metrics)
        print("Evaluation Results epoch %i"%(ck*dispIt), evalResults)



if __name__ == "__main__":
    tf.app.run()

