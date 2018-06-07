"""Convolutional neural network for exploring multi-scale strategies"""

# Tensorflow implementation

# imports 
import numpy as np
# Used for reading in images
#import cv2
import scipy.misc as misc
# used for timing 
import time

# tensorflow imports for flowing those tensors
import tensorflow as tf
#from tensorflow.contrib import learn
#from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# plotting imports
import matplotlib
import matplotlib.pyplot as plt

# user-definable flags
FLAGS = tf.app.flags.FLAGS

# User defined flags
#tf.app.flags.DEFINE_string('model','MDAC',"""Model architecture. Choices: DAC, ASPP, UNet, MDAC""")
tf.app.flags.DEFINE_boolean('restore', False,"""Restore previously trained model""")
tf.app.flags.DEFINE_integer('maxSteps', 100,"""number of epochs""")
tf.app.flags.DEFINE_integer('dispIt', 20,"""display every nth iteration""")
tf.app.flags.DEFINE_integer('batchSize', 8,"""Number of entries per minibatch""")
tf.app.flags.DEFINE_float('lR', 3e-4,"""learning rate""")
tf.app.flags.DEFINE_integer('poolStride', 2,"""display every nth iteration""")

restore = FLAGS.restore
dispIt = FLAGS.dispIt
maxSteps = FLAGS.maxSteps
batchSize = FLAGS.batchSize
lR = FLAGS.lR
poolStride = FLAGS.poolStride
myModel = "fullyCNN" #FLAGS.model

if (myModel == 'fullyCNN'):
	myModelFN = "./models/multiscale/DAC/"



# hyperparameters
mySeed = 42
dORate = 0.5
atrousdORate = 0.1

# number of kernels per layer
convDepth = 4
myBias = 0#1.0

# Image characteristics
dimY = 672#1344#336#672#imgWidth
dimX = 512#1024#256#imgHeight
myChan = 1
myOutChan = 1
# ***

kern1Size = 3
kern2Size = 3


data = tf.placeholder("float",[None, dimX,dimY,myChan], name='X')

learningRate = tf.placeholder("float",name='learningRate')

mode = tf.placeholder("bool",name="myMode")

# Define atrous conv filtes

#  convolution kernels from layer 1 to layer 2
aFilters12_a = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.1),name="a12aweights")
aFilters12_b = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.1),name="a12bweights")


# atrous convolution kernels from layer 2 to layer 3
aFilters23_a = tf.Variable(tf.random_normal([3, 3,2*convDepth,convDepth], stddev=0.1),name="a23aweights")
aFilters23_b = tf.Variable(tf.random_normal([3, 3,2*convDepth,convDepth], stddev=0.1),name="a23bweights")
aFilters23_c = tf.Variable(tf.random_normal([3, 3,2*convDepth,convDepth], stddev=0.1),name="a23cweights")

# atrous convolution kernels from layer 3 to layer 4
aFilters34_a = tf.Variable(tf.random_normal([3, 3,3*convDepth,convDepth], stddev=0.1),name="a34aweights")
aFilters34_b = tf.Variable(tf.random_normal([3, 3,3*convDepth,convDepth], stddev=0.1),name="a34bweights")
aFilters34_c = tf.Variable(tf.random_normal([3, 3,3*convDepth,convDepth], stddev=0.1),name="a34cweights")
aFilters7 = tf.Variable(tf.random_normal([3, 3,4*convDepth,1], stddev=0.1),name="a34aweights")

# For use with batch-norm
#gamma4 = tf.Variable(1.0,trainable=True)
#beta4 = tf.Variable(0.0, trainable=True)


def multiscaleNet(data,mode):
	# mode = false apply dropout
	# mode = true don't apply dropout, i.e. for evaluation/test
	#tf.image.resize_images(data,[dimX,dimY])
	inputLayer = tf.reshape(data,[-1,dimX,dimY,myChan])
	
	"""Layer 0"""
	conv0 = tf.layers.conv2d(
		inputs = inputLayer,
		filters = convDepth*2,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv0")
	pool0 = tf.nn.avg_pool(conv0,[1,poolStride,poolStride,1],	
		[1,poolStride,poolStride,1],padding="SAME")
	dropout0 = tf.layers.dropout(
		inputs = pool0,#conv1,
		rate = dORate,
		training = mode,
		name = "dropout0")#128x16

	"""Layer 1"""
	conv1 = tf.layers.conv2d(
		inputs = dropout0,
		filters = convDepth*2,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv1")
	#128x16
	pool1 = tf.nn.avg_pool(conv1,[1,poolStride,poolStride,1],	
		[1,poolStride,poolStride,1],padding="SAME")
	
	dropout1 = tf.layers.dropout(
		inputs = pool1,
		rate = dORate,
		training = mode,
		name = "dropout1")

	conv2 = tf.layers.conv2d(
		inputs = dropout1,
		filters = convDepth*2,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv2")
	pool2 = tf.nn.avg_pool(conv2,[1,poolStride,poolStride,1],	
		[1,poolStride,poolStride,1],padding="SAME")
	
	dropout2 = tf.layers.dropout(
	    inputs = pool2,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout2")#128x16
	
	conv3 = tf.layers.conv2d(
		inputs = dropout2,
		filters = convDepth*2,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv3")
	up3 = tf.image.resize_images(conv3,[int(dimX/4),int(dimY/4)])
	
	dropout3 = tf.layers.dropout(
	    inputs = up3,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout3")

	conv4 = tf.layers.conv2d(
		inputs = dropout3,
		filters = convDepth*2,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv4")
	
	up4 = tf.image.resize_images(conv4,[int(dimX/2),int(dimY/2)])


	dropout4 = tf.layers.dropout(
	    inputs = up4,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout4")
	
	conv5 = tf.layers.conv2d(
		inputs = dropout4,
		filters = convDepth*4,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, use_bias = True, 
	        bias_initializer = tf.constant_initializer(myBias),name = "conv5")
	
	#if(myModel == 'UNet'):
	up5 = tf.image.resize_images(conv4,[int(dimX),int(dimY)])
	#else:
	#	up5 = conv5

	dropout5 = tf.layers.dropout(
		inputs = up5,
		rate = dORate,
		training = mode,
		name = "dropout5")	
	
	conv6 = tf.layers.conv2d(
		inputs = dropout5,
		filters = convDepth*4,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),
		name = "conv6")

	dropout6 = tf.layers.dropout(
		inputs = conv6,
		rate = dORate/2,
		training = mode,
		name = "dropout6")

	# Linear output layer
	output = tf.nn.atrous_conv2d(dropout6,aFilters7,1,"SAME")
	"""tf.layers.conv2d(
	inputs = dropout6,
	filters = 1,
	kernel_size = [kern1Size,kern1Size],
	padding = "same",
	activation = None, 
	use_bias = True, 
	bias_initializer = tf.constant_initializer(myBias),
	name = "conv7")"""
	#output = conv7
		
	print("conv0 shape: %s"%conv0.shape)
	print("conv1 shape: %s"%conv1.shape)
	print("conv2 shape: %s"%conv2.shape)
	print("conv3 shape: %s"%conv3.shape)
	print("conv4 shape: %s"%conv4.shape)
	
	#print("mean4/var4 shape: %s/%s \n values: "%(mean4.shape,var4.shape),mean4,var4)
	#print("bnorm4 shape: %s"%bnorm4.shape)
	#print("atrous shape: %s"%atrous1.shape)
	print("conv5 shape: %s"%conv5.shape)
	print("conv6 shape: %s"%conv6.shape)
	print("conv7 shape: %s"%output.shape)
	#print("conv7 shape: %s"%conv8.shape)
	#print("conv8 shape: %s"%conv9.shape)
	
	return output

myOut = multiscaleNet(data,mode)

print("output shape",np.shape(myOut))	

# operating in autoencoder mode
loss = tf.sqrt(tf.reduce_mean(tf.pow(data - myOut, 2)))

#loss = tf.reduce_mean(targets - myOut)
trainOp = tf.train.AdamOptimizer(
	learning_rate=lR,beta1=0.9,
	beta2 = 0.999,
	epsilon=1e-08,
	use_locking=False,
	name='Adam').minimize(loss,global_step = tf.contrib.framework.get_global_step())

mySaver = tf.train.Saver()

init = tf.global_variables_initializer()
def main(unused_argv):
	#tf.reset_default_graph()
	t0 = time.time()
	with tf.Session() as sess: 
		if(restore):
			mySaver.restore(sess,tf.train.latest_checkpoint(myModelFN))
		#tf.initialize_all_variables().run() 
		sess.run(init)
		lR = 3e-5
		myX = np.load('./coelTrain2.npy')
		myVal = np.load('./coelVal2.npy')
		myMinV = np.min(myVal)
		
		myMaxV = np.max(myVal-myMinV)
		myVal = (myVal-myMinV)/(myMaxV)

		myVal = np.reshape(myVal, (myVal.shape[0],myVal.shape[1],myVal.shape[2],1))
		
		myMin = np.min(myX)
		
		myMax = np.max(myX-myMin)
		myX = (myX-myMin)/(myMax)
		myX = np.reshape(myX, (myX.shape[0],myX.shape[1],myX.shape[2],1))
		
		for i in range(maxSteps):
			for ck in range(0,len(myX),batchSize):
				input_ = myX[ck:ck+batchSize]

				sess.run(trainOp, feed_dict = {data: input_, learningRate: lR, mode: True})

			if(i% dispIt ==0):
				mySaver.save(sess,myModelFN,global_step=i)			
				inp = tf.placeholder(tf.float32)
				myMean = tf.reduce_mean(inp)
				myTemp = (sess.run(loss, feed_dict={data: input_, learningRate: lR, mode: False}))
				myLossTrain = myMean.eval(feed_dict={inp: myTemp})
				
				myTemp = (sess.run(loss, feed_dict={data: myVal, learningRate: lR, mode: False}))
				myLossVal = myMean.eval(feed_dict={inp: myTemp})
				elapsed = time.time() - t0
				print("Epoch %i training loss, validation loss: %.3e , %.3e , elapsed time: %.2f "%(i,myLossTrain,myLossVal,elapsed))

				recon = sess.run(myOut,feed_dict = {data: myVal, mode: False})
				plt.figure(figsize=(10,10))
				for ck in range(3):
					plt.subplot(3,2,2*ck+1)
					plt.title("original image")
					plt.imshow(myVal[ck,:,:,0],cmap="gray")
					plt.subplot(3,2,2*ck+2)
					plt.title("autodecoded")
					plt.imshow(recon[ck,:,:,0],cmap="gray")

				
				plt.savefig("./figs/epoch%i%s.png"%(i,myModel))		
				#plt.show()
				plt.clf()
		# Perform final evaluation
		myTest = np.load('./coelTest2.npy')
		myMinT = np.min(myTest)
	
		myMaxT = np.max(myTest-myMinT)
		myTest = (myTest-myMinT)/(myMaxT)

		myTest = np.reshape(myTest, (myTest.shape[0],myTest.shape[1],myTest.shape[2],1))
	
		inp = tf.placeholder(tf.float32)
		myMean = tf.reduce_mean(inp)
		myTemp = (sess.run(loss, feed_dict={data: input_, learningRate: lR, mode: False}))
		myLossTrain = myMean.eval(feed_dict={inp: myTemp})

		myTemp = (sess.run(loss, feed_dict={data: myVal, learningRate: lR, mode: False}))
		myLossVal = myMean.eval(feed_dict={inp: myTemp})
	
		myTemp = (sess.run(loss, feed_dict={data: myTest, learningRate: lR, mode: False}))
		myLossTest = myMean.eval(feed_dict={inp: myTemp})
		elapsed = time.time()-t0
		print("Final training loss, validation loss, test loss: %.3e , %.3e , %.3e, time elapsed: %.1f s"%(myLossTrain,myLossVal,myLossTest,elapsed))
		logFile = open("./lossLog.txt",'a')
		logFile.write("\n%s, Final training loss, validation loss, test loss: %.3e , %.3e , %.3e, time elapsed: %.1f s"%(myModel, myLossTrain,myLossVal,myLossTest,elapsed))
		recon = sess.run(myOut,feed_dict = {data: myTest, mode: False})
		print(np.shape(myTest))
		print(np.shape(recon))
		for ck in range(9):
			plt.figure(figsize=(10,5))
			plt.subplot(1,2,1)
			plt.title("original image")
			plt.imshow(myTest[ck,:,:,0],cmap="gray")
			plt.subplot(1,2,2)
			plt.title("autodecoded w/  %s"%(myModel))
			plt.imshow(recon[ck,:,:,0],cmap="gray")

			plt.savefig("./figs/testAE%i%s.png"%(ck,myModel))
		np.save(('./output/coelTestData%s.npy'%(myModel)),recon)
		plt.clf()

	print("finished .. . .")


if __name__ == "__main__":
    tf.app.run()
