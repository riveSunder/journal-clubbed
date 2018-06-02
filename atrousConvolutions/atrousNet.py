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
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# plotting imports
import matplotlib
import matplotlib.pyplot as plt

# Determine whether we are using atrous convolutions, u-net architecture, neither, or both


# user-definable flags
FLAGS = tf.app.flags.FLAGS

# User defined flags
tf.app.flags.DEFINE_boolean('useAtrous', False,"""Use dilated convolutions?""")
tf.app.flags.DEFINE_boolean('useUNet', False,"""Use skip connections?""")
tf.app.flags.DEFINE_boolean('restore', False,"""Restore previously trained model""")
tf.app.flags.DEFINE_integer('maxSteps', 1,"""number of epochs""")
tf.app.flags.DEFINE_integer('dispIt', 20,"""display every nth iteration""")
tf.app.flags.DEFINE_integer('batchSize', 4,"""Number of entries per minibatch""")

restore = FLAGS.restore
useAtrous = FLAGS.useAtrous
useUNet = FLAGS.useUNet
dispIt = FLAGS.dispIt
maxSteps = FLAGS.maxSteps
batchSize = FLAGS.batchSize


if (useAtrous and useUNet):
	myModel = "./models/holesAndSkips/"
elif (useAtrous):
	myModel = "./models/justHoles/"
elif (useUNet):
	myModel = "./models/justSkips/"
else:
	myModel = "./models/noHolesNoSkips/"




# hyperparameters
mySeed = 1337

dORate = 0.55
lR = 3e-4
# number of kernels per layer
convDepth = 4
myBias = 0.1#1.0
#dimX = 482
#dimY = 646
# Image characteristics
dimY = 1344#336#672#imgWidth
dimX = 1024#256#imgHeight
myChan = 1
myOutChan = 1
# ***
pool1Size = 1#2   
pool2Size = 1
kern1Size = 3
kern2Size = 3


data = tf.placeholder("float",[None, dimX,dimY,myChan], name='X')

learningRate = tf.placeholder("float",name='learningRate')

mode = tf.placeholder("bool",name="myMode")

# Define atrous conv filtes
a1Filters = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.35),name="a1weights")
a2Filters = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.35),name="a2weights")
a3Filters = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.35),name="a3weights")
a4Filters = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.35),name="a4weights")
a8Filters = tf.Variable(tf.random_normal([3, 3,convDepth,convDepth], stddev=0.35),name="a8weights")

# bias for atrous layer
#ba = tf.Variable(myBias,trainable=True)

def atrousCNN(data,mode):
	# mode = false apply dropout
	# mode = true don't apply dropout, i.e. for evaluation/test
	
	inputLayer = tf.reshape(data,[-1,dimX,dimY,myChan])
	
	"""Layer 0"""
	conv0 = tf.layers.conv2d(
		inputs = inputLayer,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, use_bias = True, bias_initializer = tf.constant_initializer(myBias),name = "conv0")
	pool0 = tf.nn.avg_pool(conv0,[1,2,2,1],[1,2,2,1],padding="SAME")
	dropout0 = tf.layers.dropout(
	    inputs = pool0,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout0")#128x16

	"""Layer 1"""
	conv1 = tf.layers.conv2d(
		inputs = dropout0,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, use_bias = True, bias_initializer = tf.constant_initializer(myBias),name = "conv1")
	#128x16
	#tf.nn.avg_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME")
	dropout1 = tf.layers.dropout(
	    inputs = conv1,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout1")
	"""Layer 1.1"""
	conv2 = tf.layers.conv2d(
		inputs = dropout1,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
        use_bias = True, bias_initializer = tf.constant_initializer(myBias),name = "conv11")
	pool2 = tf.nn.avg_pool(conv2,[1,2,2,1],[1,2,2,1],padding="SAME")

	dropout2 = tf.layers.dropout(
	    inputs = pool2,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout11")#128x16
    
	conv3 = tf.layers.conv2d(
		inputs = dropout2,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, use_bias = True, 
        bias_initializer = tf.constant_initializer(myBias),name = "conv12")
	dropout3 = tf.layers.dropout(
	    inputs = conv3,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout12")#128x16
    
	conv4 = tf.layers.conv2d(
		inputs = dropout3,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
        use_bias = True, bias_initializer = tf.constant_initializer(myBias),name = "conv13")
	dropout4 = tf.layers.dropout(
	    inputs = conv4,#conv1,
	    rate = dORate,
	    training = mode,
	    name = "dropout13")#128x16    

    
	atrous1 = tf.nn.relu(tf.nn.atrous_conv2d(dropout4,a1Filters,1,"SAME",name='atrous1'))

	if(useAtrous):
		"""Parallel atrous convolutions"""
		atrous2 = tf.nn.relu(tf.nn.atrous_conv2d(dropout4,a2Filters,2,"SAME",name='atrous2'))
		atrous3 = tf.nn.relu(tf.nn.atrous_conv2d(dropout4,a3Filters,3,"SAME",name='atrous3'))
		atrous4 = tf.nn.relu(tf.nn.atrous_conv2d(dropout4,a4Filters,4,"SAME",name='atrous4'))
		atrous8 = tf.nn.relu(tf.nn.atrous_conv2d(dropout4,a8Filters,8,"SAME",name='atrous8'))
	
			
		dropout5 = tf.layers.dropout(
			inputs = tf.concat([atrous1,atrous2,atrous3,atrous4,atrous8],3),
			rate = dORate,
			training = mode,
			name = "dropout5")
	else:
		dropout5 = tf.layers.dropout(
			inputs = atrous1, #tf.concat([atrous1,atrous2,atrous4,atrous8],3),
			rate = dORate,
			training = mode,
			name = "dropout5")

	conv6 = tf.layers.conv2d(
		inputs = dropout5,
		filters = convDepth,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = tf.nn.relu, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),name = "conv6")

	res6 =	tf.image.resize_images(conv6,[int(dimX),int(dimY)])	    
	if(useUNet):

		"""Include skip connection (U-Net architecture) from conv2"""
		dropout6u = tf.layers.dropout(
			inputs = tf.concat([conv2,res6],3),
			rate = dORate,
			training = mode,
			name = "dropout4")
		conv7 = tf.layers.conv2d(
			inputs = dropout6u,
			filters = convDepth,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),name = "conv5")	
	else:
		dropout6 = tf.layers.dropout(
			inputs = res6,
			rate = dORate,
			training = mode,
			name = "dropout4")
		conv7 = tf.layers.conv2d(
			inputs = dropout6,
			filters = convDepth,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),name = "conv5")
	
	#res7 =	tf.image.resize_images(conv7,[int(dimX),int(dimY)])	    
	if(useUNet):
		"""include skip connection to conv0 (U-Net)"""
		dropout7u = tf.layers.dropout(
			inputs = tf.concat([conv0,res7],3),
			rate = dORate,
			training = mode,
			name = "dropout7u")#128x16
		conv8 = tf.layers.conv2d(
			inputs = dropout7u,
			filters = convDepth,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),name = "conv8")
		dropout8 = tf.layers.dropout(
			inputs = conv8,
			rate = dORate,
			training = mode,
			name = "dropout8")	

	else:
		dropout7 = tf.layers.dropout(
			inputs = conv7,
			rate = dORate,
			training = mode,
			name = "dropout7")	

		conv8 = tf.layers.conv2d(
			inputs = dropout7,
			filters = convDepth,
			kernel_size = [kern1Size,kern1Size],
			padding = "same",
			activation = tf.nn.relu, 
			use_bias = True, 
			bias_initializer = tf.constant_initializer(myBias),name = "conv8")
		dropout8 = tf.layers.dropout(
			inputs = conv8,
			rate = dORate,
			training = mode,
			name = "dropout8")	

	conv9 = tf.layers.conv2d(
		inputs = dropout8,
		filters = 1,
		kernel_size = [kern1Size,kern1Size],
		padding = "same",
		activation = None, 
		use_bias = True, 
		bias_initializer = tf.constant_initializer(myBias),name = "conv9")
		

	myOutput = conv9
	return myOutput

myOut = atrousCNN(data,mode)

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
	t0 = time.time()
	#tf.reset_default_graph()
	with tf.Session() as sess: 
		if(restore):
			mySaver.restore(sess,tf.train.latest_checkpoint(myModel))
	
		#tf.initialize_all_variables().run() 
		sess.run(init)
		lR = 3e-5
		myX = np.load('./coelTrain.npy')
		myVal = np.load('./coelVal.npy')
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
				mySaver.save(sess,myModel,global_step=i)			
				inp = tf.placeholder(tf.float32)
				myMean = tf.reduce_mean(inp)
				myTemp = (sess.run(loss, feed_dict={data: input_, learningRate: lR, mode: False}))
				myLossTrain = myMean.eval(feed_dict={inp: myTemp})
				
				myTemp = (sess.run(loss, feed_dict={data: myVal, learningRate: lR, mode: False}))
				myLossVal = myMean.eval(feed_dict={inp: myTemp})
				elapsed = time.time()-t0
				print("Epoch %i training loss, validation loss: %.3e , %.3e, time elapsed: %.1f s"%(i,myLossTrain,myLossVal,elapsed))

				recon = sess.run(myOut,feed_dict = {data: myVal, mode: False})
				plt.figure(figsize=(10,10))
				for ck in range(3):
					plt.subplot(3,2,2*ck+1)
					plt.title("original image")
					plt.imshow(myVal[ck,:,:,0],cmap="gray")
					plt.subplot(3,2,2*ck+2)
					plt.title("autodecoded")
					plt.imshow(recon[ck,:,:,0],cmap="gray")

				
				plt.savefig("./figs/epoch%iAtrous%sUNet%s.png"%(i,useAtrous,useUNet))
				plt.clf()
		# Perform final evaluation
		myTest = np.load('./coelTest.npy')
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
		logFile.write("\nAtrous: %s, UNet: %s, Final training loss, validation loss, test loss: %.3e , %.3e , %.3e, time elapsed: %.1f s"%(useAtrous,useUNet, myLossTrain,myLossVal,myLossTest,elapsed))
		recon = sess.run(myOut,feed_dict = {data: myTest, mode: False})
		print(np.shape(myTest))
		print(np.shape(recon))
		for ck in range(9):
			plt.figure(figsize=(10,5))
			plt.subplot(1,2,1)
			plt.title("original image")
			plt.imshow(myTest[ck,:,:,0],cmap="gray")
			plt.subplot(1,2,2)
			plt.title("autodecoded w/ Atrous: %s and UNet: %s"%(str(useAtrous),str(useUNet)))
			plt.imshow(recon[ck,:,:,0],cmap="gray")

			plt.savefig("./figs/testAE%iAtrous%sUNet%s.png"%(ck,str(useAtrous),str(useUNet)))
		np.save('coelTestDataAtrous%sUNet%s.npy'%(str(useAtrous),str(useUNet)),recon)
		plt.clf()

	print("finished .. . .")


if __name__ == "__main__":
    tf.app.run()
