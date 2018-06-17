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
tf.app.flags.DEFINE_boolean('dispFigs', True,"""Whether to save figures""")
tf.app.flags.DEFINE_boolean('sixLayers', False,"""Slightly deeper conv-net""")

dispFigs = FLAGS.dispFigs
restore = FLAGS.restore
dispIt = FLAGS.dispIt
maxSteps = FLAGS.maxSteps
batchSize = FLAGS.batchSize
lR = FLAGS.lR
poolStride = FLAGS.poolStride
myModel = "fullyCNN" #FLAGS.model

sixLayers = FLAGS.sixLayers


if (myModel == 'fullyCNN'):
	myModelFN = "./models/multiscale/vSimpleFCNNSeg6Layers%s/"%sixLayers



# hyperparameters
mySeed = 42
dORate = 0.5
atrousdORate = 0.1

# number of kernels per layer
convDepth = 8
myBias = 0#1.0

# Image characteristics
myX = np.load("../datasets/epflEM/epflX.npy")
dimY = myX.shape[1]#256#672#1344#336#672#imgWidth
dimX = myX.shape[2]# 256#512#1024#256#imgHeight
myChan = 1
myOutChan = 2
# ***

kern1Size = 5
kern2Size = 3

data = tf.placeholder("float",[None, dimX,dimY,myChan], name='X')

targets = tf.placeholder("int32",[None, dimX, dimY], name='Y')
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


def multiscaleNet(data,targets,mode):
	# mode = false apply dropout
	# mode = true don't apply dropout, i.e. for evaluation/test
	#tf.image.resize_images(data,[dimX,dimY])
	inputLayer = tf.reshape(data,[-1,dimX,dimY,myChan])
	
	"""Layer 0"""
	conv0 = tf.layers.conv2d(
		inputs = inputLayer,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.leaky_relu, 
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
		activation = tf.nn.leaky_relu, 
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
	if(sixLayers):
		conv2 = tf.layers.conv2d(
			inputs = dropout1,
			filters = convDepth*4,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.leaky_relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),
			name = "conv2")
		dropout2 = tf.layers.dropout(
			inputs = conv2,
			rate = dORate,
			training = mode,
			name = "dropout2")

		conv3 = tf.layers.conv2d(
			inputs = dropout2,
			filters = convDepth*4,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.leaky_relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),
			name = "conv3")
		up3 = tf.image.resize_images(conv3,[int(dimX/2),int(dimY/2)])
		dropout3 = tf.layers.dropout(
			inputs = up3,
			rate = dORate,
			training = mode,
			name = "dropout3")
	
		"""
		dropout4 = tf.layers.dropout(
			inputs = up4,#conv1,
			rate = dORate,
			training = mode,
			name = "dropout4")
		"""
		conv4 = tf.layers.conv2d(
				inputs = dropout3,
				filters = convDepth*2,
				kernel_size = [kern1Size,kern1Size],
				padding = "same",
				activation = tf.nn.leaky_relu, 
				use_bias = True, 
				bias_initializer = tf.constant_initializer(myBias),
				name = "conv4")
	else:
		conv4 = tf.layers.conv2d(
			inputs = dropout1,
			filters = convDepth*2,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.leaky_relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),
			name = "conv4")
	
	up4 = tf.image.resize_images(conv4,[int(dimX),int(dimY)])


	dropout4 = tf.layers.dropout(
	    inputs = up4,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout4")
	
	conv5 = tf.layers.conv2d(
		inputs = dropout4,
		filters = myOutChan, #convDepth*4,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = None, #tf.nn.leaky_relu, 
		use_bias = True, 
	    bias_initializer = tf.constant_initializer(myBias)
		)
	
	output = conv5
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
	if(sixLayers):
		print("conv2 shape: %s"%conv2.shape)
		print("conv3 shape: %s"%conv3.shape)
	print("conv4 shape: %s"%conv4.shape)
	
	#print("mean4/var4 shape: %s/%s \n values: "%(mean4.shape,var4.shape),mean4,var4)
	#print("bnorm4 shape: %s"%bnorm4.shape)
	#print("atrous shape: %s"%atrous1.shape)
	print("conv5 shape: %s"%conv5.shape)
	#print("conv7 shape: %s"%conv8.shape)
	#print("conv8 shape: %s"%conv9.shape)
	
	return output

myOut = multiscaleNet(data,targets,mode)

print("output shape",np.shape(myOut))	

# operating in autoencoder mode
lossMSE = tf.sqrt(tf.reduce_mean(tf.pow(data - myOut, 2)))


#loss = tf.reduce_mean(targets - myOut)
trainOpAE = tf.train.AdamOptimizer(
	learning_rate=lR,beta1=0.9,
	beta2 = 0.999,
	epsilon=1e-08,
	use_locking=False,
	name='Adam').minimize(lossMSE,global_step = tf.contrib.framework.get_global_step())

# segmentation loss and train op


interimLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(indices = tf.cast(targets,tf.int32),depth=myOutChan),logits=myOut)

#loss = tf.reduce_mean(interimLoss)
reshapedLogits = tf.reshape(myOut,[-1,myOutChan])
reshapedLabels = tf.reshape(targets,[-1])
reshapedLabels = tf.one_hot(indices = tf.cast(reshapedLabels,tf.int32),depth=myOutChan)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=reshapedLabels,logits=reshapedLogits)#,pos_weight=2)


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
		sess.run(init)	
		if(restore):
			print("restoring model from disk: ",myModelFN)		
			mySaver.restore(sess,tf.train.latest_checkpoint(myModelFN))
		#tf.initialize_all_variables().run() 
		
		#lR = 3e-5
		if(0):
			myX = np.array(np.load('../datasets/kaggleNuclei/X256.npy'))
			myY = np.array(np.load('../datasets/kaggleNuclei/Y256.npy'))
			myVal = myX[0:20,:,:,0]
			myYVal = myY[0:20,...]
			myX = myX[20:len(myX),:,:,0]
			myY = myY[20:len(myY),...]
			print(myX.shape,myY.shape,myVal.shape,myYVal.shape)
		elif(1):
			myX = np.load("../datasets/epflEM/epflX.npy")
			myVal = np.load("../datasets/epflEM/epflXVal.npy")
			myY = np.load("../datasets/epflEM/epflY.npy")
			myYVal = np.load("../datasets/epflEM/epflYVal.npy")
			#myY = np.reshape(myY,(len(myY),dimX,dimY,1))
			#myYVal = np.reshape(myYVal,(len(myYVal),dimX,dimY,1))
		elif(1):
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
	
		#myTgts = np.zeros((len(myY),dimX,dimY,2))	
		#myTgtsVal = np.zeros((len(myYVal,dimX,dimY,2))
		myY = myY / np.max(myY)
		myYVal = myYVal/np.max(myYVal)
	
		for i in range(750,maxSteps):
			for ck in range(0,len(myX),batchSize):
				input_ = myX[ck:ck+batchSize,:,:,:]
				targets_ = myY[ck:ck+batchSize,:,:]

				sess.run(trainOp, feed_dict = {data: input_, targets:targets_, learningRate: lR, mode: True})
				#print("post training call marker")
			if(i% dispIt ==0):
				mySaver.save(sess,myModelFN,global_step=i)			
				
				ck = int(len(myX)*np.random.random()-1)				
				input_ = myX[ck:ck+batchSize,:,:,:]
				targets_ = myY[ck:ck+batchSize,:,:]

				inp = tf.placeholder(tf.float32)
				myMean = tf.reduce_mean(inp)
				myTemp = (sess.run(loss, feed_dict={data: input_, targets:targets_, learningRate: lR, mode: False}))
				myLossTrain = myMean.eval(feed_dict={inp: myTemp})
				
				myTemp = (sess.run(loss, feed_dict={data: myVal, targets:myYVal, learningRate: lR, mode: False}))
				myLossVal = myMean.eval(feed_dict={inp: myTemp})

				elapsed = time.time() - t0
				print("Epoch %i training loss, validation loss: %.3e , %.3e , elapsed time: %.2f "%(i,myLossTrain,myLossVal,elapsed))
				if(1):
					trainLogFile = open("./trainLogs/verySimpleSegTrainingLog.txt",'a')
					trainLogFile.write("%i, %.3e, %.3e, %.3e\n"%(i,myLossTrain,myLossVal,elapsed))
					trainLogFile.close()
				if(dispFigs):
					recon = sess.run(myOut,feed_dict = {data: myVal, targets:myYVal, mode: False})
					#myDispLabels = np.argmax(myYVal,axis=3)
					myReconLabels = np.argmax(recon,axis=3)
					plt.figure(figsize=(12,12))
					for ck in range(3):
						plt.subplot(3,3,3*ck+1)
						#plt.title("original image")
						plt.imshow(myVal[ck+15,:,:,0],cmap="gray")
						#plt.imshow(recon[ck,:,:,0],cmap="gray")
						plt.subplot(3,3,3*ck+2)
						plt.title("Segmentation")
						plt.imshow(myReconLabels[ck+10,:,:],cmap="gray")
						#myProb = np.exp(recon[15+ck,:,:,1])/(np.exp(recon[15+ck,:,:,1])+np.exp(recon[15+ck,:,:,0]))
						#print(myProb.shape)						
						#plt.imshow(myProb)
						#plt.imshow(myReconLabels[15+ck,:,:],cmap="gray")
						#plt.colorbar()
						plt.subplot(3,3,3*ck+3)
						plt.title("target")
						plt.imshow(myYVal[ck+15,:,:],cmap="gray")
						#plt.colorbar()


				
					plt.savefig("./figs/vSimpleEpoch%i%s.png"%(i,myModel))		
					#plt.show()
					plt.clf()
	recon = sess.run(myOut,feed_dict = {data: myVal, targets:myYVal, mode: False})
	np.save("./output/verySimpleFCNNSegVal.npy",recon)	
	print("finished .. . .")


if __name__ == "__main__":
    tf.app.run()
