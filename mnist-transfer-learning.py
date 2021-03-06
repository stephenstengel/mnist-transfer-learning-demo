#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  mnist-transfer-learning.py
#  
#  2022 Stephen Stengel <stephen.stengel@cwu.edu>

#A test program to see how transfer learning works.
#Tutorial sources:
# ~ https://keras.io/examples/vision/image_classification_from_scratch/
# ~ https://keras.io/guides/transfer_learning/
# ~ https://towardsdatascience.com/transfer-learning-using-pre-trained-alexnet-model-and-fashion-mnist-43898c2966fb?gi=1f9cc1728578
# ~ https://www.kaggle.com/code/muerbingsha/mnist-vgg19/notebook

## ! Training on mnist and then using those weights in a fashion_mnist
## ! model would be a way easier tutorial. or cifar10

print("Running imports...")

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist		#images of digits.
from keras.datasets import fashion_mnist  #images of clothes
# ~ from keras.datasets import cifar10    #small images
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split

print("Done!")

IS_GLOBAL_PRINTING_ON = False

HELPFILE_PATH = "helpfile"

GLOBAL_DEFAULT_SLICE = 2000
GLOBAL_DEFAULT_EPOCHS = 100

def main(args):
	print("Hello!")

	#If I check args before imports, it will be faster.
	#Just save returns to their globals.
	sliceNum, epochsMnist = checkArgs(args)
	
	if IS_GLOBAL_PRINTING_ON:
		print("global printing true")
		print("sliceNum: %d" % sliceNum)
		print("epochsNum: %d" % epochsMnist)

	dFolder = "./digits/"
	os.system("mkdir -p " + dFolder)
	fFolder = "./fashion/"
	os.system("mkdir -p " + fFolder)
	
	# ~ preamble()
	# ~ epochsCat = 10
	# ~ xceptCatDog(epochsCat)
	
	xceptionOnMnistExample(epochsMnist, mnist, dFolder, sliceNum)
	xceptionOnMnistExample(epochsMnist, fashion_mnist, fFolder, sliceNum)
	# ~ xceptionOnMnistExample(epochsMnist, cifar10, cFolder, sliceNum)

	return 0


#Check if user wants help
def checkArgs(args):
	helpList = ["help", "-help", "--help", "-h", "--h", "wtf", "-wtf", "--wtf"]
	argLen = len(args)
	if argLen >= 1:
		for a in args:
			if str(a).lower() in helpList:
				printFile(HELPFILE_PATH)
				sys.exit(0)
	if argLen == 1:
		return GLOBAL_DEFAULT_SLICE, GLOBAL_DEFAULT_EPOCHS
	if argLen == 2:
		print("bad input")
		printFile(HELPFILE_PATH)
		sys.exit(-1)
	if argLen == 3:
		theSlice = int(sys.argv[1])
		theEpochs = int(sys.argv[2])
		return theSlice, theEpochs
	if argLen == 4:
		theSlice = int(sys.argv[1])
		theEpochs = int(sys.argv[2])
		global IS_GLOBAL_PRINTING_ON
		IS_GLOBAL_PRINTING_ON = True
		return theSlice, theEpochs
	if argLen > 4:
		print("bad input")
		printFile(HELPFILE_PATH)
		sys.exit(-1)
		
		
#Prints a text file to screen
def printFile(myFilePath):
	with open(myFilePath, "r") as helpfile:
		for line in helpfile:
			print(line, end = "")


#Example of using xception network with imagenet weights to do transfer
#learning on mnist. I'll cut down the classes in MNIST to just two to 
#more closely match the tutorial I'm following and to match the problem
#that we will be solving for the project. HEHE
def xceptionOnMnistExample(epochsMnist, myDataset, tmpFolder, sliceNum):
	print("Creating datasets...")
	
	train_ds, validation_ds, test_ds = readDataset(myDataset, sliceNum)
	
	print("Done!")
	
	if IS_GLOBAL_PRINTING_ON:
		print("Showing some images from the dataset...")
		printSomeOfDataset(train_ds)
		printSomeOfDataset(validation_ds)
		printSomeOfDataset(test_ds)
		print("Done!")

	print("Making the model...")
	model = makeTheModel()
	print("Done!")


	print("Training top layer...")
	
	epochs = epochsMnist
	if IS_GLOBAL_PRINTING_ON:
		print("train_ds: " + str(train_ds))
	myHistory = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
	
	print("Done!")
	
	predictStuff(model, test_ds)

	performEvaluation(myHistory, tmpFolder, model, test_ds)
	
	print("HORAAAAAYYYYY")
	


def predictStuff(model, the_ds ):
	predictions = model.predict(the_ds)
	score = predictions[0]
	if IS_GLOBAL_PRINTING_ON:
		print("predictions: " + str(predictions))
	print("score: " + str(score))
	#This is buggy. Maybe was made for a different type of score.
	print(
		"This image is %.2f percent thing and %.2f percent thing2."
		% (100 * (1 - score), 100 * score)
	)


def makeTheModel():
	base_model = keras.applications.Xception(
		weights="imagenet",  # Load weights pre-trained on ImageNet.
		input_shape=(150, 150, 3),
		include_top=False,
	)  # Do not include the ImageNet classifier at the top.
	
	# Freeze the base_model
	base_model.trainable = False
	
	# Create new layers on top of the old model
	inputs = keras.Input(shape=(150, 150, 3))
	x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
	
	# Pre-trained Xception weights requires that input be scaled
	# from (0, 255) to a range of (-1., +1.), the rescaling layer
	# outputs: `(inputs * scale) + offset`
	scale_layer = keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1)
	x = scale_layer(x)
	
	# The base model contains batchnorm layers. We want to keep them in inference mode
	# when we unfreeze the base model for fine-tuning, so we make sure that the
	# base_model is running in inference mode here.
	x = base_model(x, training=False)
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)
	
	model.summary()
	
	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=[keras.metrics.BinaryAccuracy()],
	)
	
	return model

#prints a few images from a loaded dataset.
def printSomeOfDataset(myDataset):
	print("Looking at a few of the images...")
	plt.figure(figsize=(10, 10))
	for images, labels in myDataset.take(1):
		for i in tqdm(range(9)):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("float32"))
			plt.title(int(labels[i]))
			plt.axis("off")
	plt.show()

#This reads the specified dataset into memory.
#Try: mnist, fashion_mnist, cifar10
def readDataset( myDataset, sliceNum ):
	print("Start reading the data ...")
	# Get the time
	StartTime = time.time()
	(x_train, y_train), (x_test, y_test) = myDataset.load_data()
	
	#I remember shuffling pairs of numpy arrays once using a random order list and zip() or something like that.
	#Can't shuffle here because the arrays are nonwriteable. 
	# ~ rng = np.random.default_rng()
	# ~ rng.shuffle(x_train)
	# ~ rng.shuffle(y_train)
	# ~ rng.shuffle(x_test)
	# ~ rng.shuffle(y_test)
	
	#Full set is too much memory with these LISTS coming up below.
	#Currently slice of 30000 is a bit under 15GB
	cutNum = sliceNum
	x_train = x_train[:cutNum]
	y_train = y_train[:cutNum]
	x_test = x_test[:cutNum]
	y_test = y_test[:cutNum]
	#######
	
	#make validation set from training data.
	splitDecimal = 0.8
	x_train, x_val = valTrainSplit(x_train, splitDecimal)
	y_train, y_val = valTrainSplit(y_train, splitDecimal)
	
	if IS_GLOBAL_PRINTING_ON:
		print("SHAPES:...")
		print(x_train.shape,y_train.shape)
		print(x_test.shape, y_test.shape)
		print(x_val.shape, y_val.shape)

	#I just picked these things at random.
	ANKLE_BOOT = 9
	T_SHIRT = 0
	firstClass = ANKLE_BOOT
	secondClass = T_SHIRT
	
	#This is very slow because I put them in lists before converting to numpy arrays.
	#TODO: remember how to do that numpy extend thing. ##################################################!
	size = (150, 150)
	x_train, y_train = keepTwoClasses(x_train, y_train, firstClass, secondClass, size)
	x_test, y_test = keepTwoClasses(x_test, y_test, firstClass, secondClass, size)
	x_val, y_val = keepTwoClasses(x_val, y_val, firstClass, secondClass, size)
	
	if IS_GLOBAL_PRINTING_ON:
		print("SHAPES after selection of two classes...")
		print(x_train.shape,y_train.shape)
		print(x_test.shape, y_test.shape)
		print(x_val.shape, y_val.shape)
	
	
	## ! NOTE THE NAME CHANGE ! ##
	# x_test, y_test becomes val_ds
	# x_val, y_val becomes realval_ds
	# I can change these names to be more consistent with the return order later.
	train_ds     = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	val_ds       = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	realval_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
	

	BATCH_SIZE = 64
	train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
	val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
	realval_ds = realval_ds.batch(BATCH_SIZE, drop_remainder=False)
	
	
	train_ds = train_ds.prefetch(buffer_size=32)
	val_ds = val_ds.prefetch(buffer_size=32)
	realval_ds = realval_ds.prefetch(buffer_size=32)
	
	#Get the time
	EndTime = time.time()
	print("Elapsed time to read dataset into memory: ", EndTime - StartTime)
	
	print("WORKSOFAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	
	#realval_ds is the dataset for validation. val_ds is actually the test set. change names later.
	return train_ds, realval_ds, val_ds


#Keeps only two classes out of a total dataset.
#In need of optimization.
def keepTwoClasses(x_train, y_train, firstClass, secondClass, size):
	newxtrain = []
	newytrain = []
	for i in tqdm(range(len(x_train))):
		if y_train[i] == firstClass or y_train[i] == secondClass:
			# ~ print("ankle boot lol")
			newxtrain.append(gray2rgb(resize(x_train[i], size)) )
			newytrain.append(y_train[i])
	x_train = np.asarray(newxtrain)
	y_train = np.asarray(newytrain)
	
	return x_train, y_train


#Split a validation set from the training set.
#There is already a test set created by the mnist loading function.
def valTrainSplit(x_train, splitDecimal):
	sliceidx = int(len(x_train) * splitDecimal)
	x_val = x_train[sliceidx:]
	x_train = x_train[:sliceidx]
	
	return x_train, x_val


#Example of using a pretrained network with weights, adding a bit to the
#model, and training on a new dataset. This example loads weights from
#imagenet and puts them in a xception network. Then it adds a few
#layers, changes to create binary output, and runs against the
#cats_vs_dogs dataset.
def xceptCatDog(epochsCat):
	#Starting by importing the dataset manually because the first tutorial just doesn't compile.
	#Using this other tutorial: https://keras.io/examples/vision/image_classification_from_scratch/
	
	# ~ imagesTopFolderName = "PetImages"
	imagesTopFolderName = "shorter-pet-images"
	
	print("Cleaning out images that the tutorial doesn't like.")
	num_skipped = 0
	for folder_name in ("Cat", "Dog"):
		folder_path = os.path.join(imagesTopFolderName, folder_name)
		for fname in tqdm(os.listdir(folder_path)):
			fpath = os.path.join(folder_path, fname)
			try:
				fobj = open(fpath, "rb")
				is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
			finally:
				fobj.close()
	
			if not is_jfif:
				num_skipped += 1
				# Delete corrupted image
				os.remove(fpath)
	
	print("Deleted %d images" % num_skipped)
	
	print("Creating datasets...")
	
	image_size = (180, 180)
	batch_size = 32
	
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		imagesTopFolderName,
		validation_split=0.2,
		subset="training",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		imagesTopFolderName,
		validation_split=0.2,
		subset="validation",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	
	print("Done!")
	print("train_ds: " + str(train_ds) + " ##################################################")
	
	# ~ print("Looking at a few of the images...")
	# ~ plt.figure(figsize=(10, 10))
	# ~ for images, labels in train_ds.take(1):
		# ~ for i in tqdm(range(9)):
			# ~ ax = plt.subplot(3, 3, i + 1)
			# ~ plt.imshow(images[i].numpy().astype("uint8"))
			# ~ plt.title(int(labels[i]))
			# ~ plt.axis("off")
	# ~ plt.show() #THis line is needed for the pictures to actually show.
	# ~ print("Done!")
	
	print("Viewing and Augmenting data...")
	#Need to use experimental tag because my conda instaled tensorflow 2.4
	# ~ data_augmentation = keras.Sequential(
		# ~ [
			# ~ layers.experimental.preprocessing.RandomFlip("horizontal"),
			# ~ layers.experimental.preprocessing.RandomRotation(0.1),
		# ~ ]
	# ~ )
	
	#comment to faster lol
	# ~ plt.figure(figsize=(10, 10))
	# ~ for images, _ in train_ds.take(1):
		# ~ for i in tqdm(range(9)):
			# ~ augmented_images = data_augmentation(images)
			# ~ ax = plt.subplot(3, 3, i + 1)
			# ~ plt.imshow(augmented_images[0].numpy().astype("uint8"))
			# ~ plt.axis("off")
	# ~ plt.show()
	
	print("Done!")
	
	##### Back to the original tutorial ######
	print("Preprocessing...")
	# ~ train_ds = train_ds.prefetch(buffer_size=32)
	# ~ val_ds = val_ds.prefetch(buffer_size=32)
	
	size = (150, 150)
	print("train_ds before resize: " + str(train_ds))
	train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
	validation_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y)) #RENAME val_ds

	####### This makes the shapes wonky and causes crash.
	# ~ batch_size = 32
	# ~ train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
	# ~ validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
	
	#This simpler way works fine it seems. (I haven't gotten past one epoch yet. I need o pare down the data still.
	train_ds = train_ds.prefetch(buffer_size=32)
	validation_ds = validation_ds.prefetch(buffer_size=32)
	
	#test
	# ~ data_augmentation = keras.Sequential(
		# ~ [layers.experimental.preprocessing.RandomFlip("horizontal"), layers.experimental.preprocessing.RandomRotation(0.1),]
	# ~ )
	# ~ for images, labels in train_ds.take(1):
		# ~ plt.figure(figsize=(10, 10))
		# ~ first_image = images[0]
		# ~ for i in range(9):
			# ~ ax = plt.subplot(3, 3, i + 1)
			# ~ augmented_image = data_augmentation(
				# ~ tf.expand_dims(first_image, 0), training=True
			# ~ )
			# ~ plt.imshow(augmented_image[0].numpy().astype("int32"))
			# ~ plt.title(int(labels[0]))
			# ~ plt.axis("off")	
	# ~ plt.show()
	
	
	
	print("Done!")
	print("Making the model...")
	base_model = keras.applications.Xception(
		weights="imagenet",  # Load weights pre-trained on ImageNet.
		input_shape=(150, 150, 3),
		include_top=False,
	)  # Do not include the ImageNet classifier at the top.
	
	# Freeze the base_model
	base_model.trainable = False
	
	# Create new model on top
	inputs = keras.Input(shape=(150, 150, 3))
	# ~ x = data_augmentation(inputs)  # Apply random data augmentation
	# ~ x = inputs
	x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
	
	# Pre-trained Xception weights requires that input be scaled
	# from (0, 255) to a range of (-1., +1.), the rescaling layer
	# outputs: `(inputs * scale) + offset`
	scale_layer = keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1)
	x = scale_layer(x)
	
	# The base model contains batchnorm layers. We want to keep them in inference mode
	# when we unfreeze the base model for fine-tuning, so we make sure that the
	# base_model is running in inference mode here.
	x = base_model(x, training=False)
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)
	
	model.summary()
		
	
	
	print("Done!")
	
	
	#train
	print("Training top layer...")
	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=[keras.metrics.BinaryAccuracy()],
	)
	
	# ~ epochs = 20
	epochs = epochsCat
	print("train_ds: " + str(train_ds))
	# ~ print("train_ds shape: " + str(train_ds.shape))
	model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
	
	print("Done!")
	image_size = (150, 150)
	img = keras.preprocessing.image.load_img(
		"PetImages/Cat/6779.jpg", target_size=image_size
	)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)  # Create batch axis
	
	predictions = model.predict(img_array)
	score = predictions[0]
	print("predictions: " + str(predictions))
	print("score: " + str(score))
	print(
		"This image is %.2f percent cat and %.2f percent dog."
		% (100 * (1 - score), 100 * score)
	)
	

#an xception model from a tutorial-- UNUSED
def make_modelUNUSED(input_shape, num_classes):
	inputs = keras.Input(shape=input_shape)
	# Image augmentation block
	# ~ x = data_augmentation(inputs) #BUGGY
	
	#convert to functional or whatever.
	x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
	# ~ x = layers.experimental.preprocessing.RandomRotation(0.1)(x) #Buggy
	
	# Entry block
	x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
	x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	x = layers.Conv2D(64, 3, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	previous_block_activation = x  # Set aside residual

	for size in [128, 256, 512, 728]:
		x = layers.Activation("relu")(x)
		x = layers.SeparableConv2D(size, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)

		x = layers.Activation("relu")(x)
		x = layers.SeparableConv2D(size, 3, padding="same")(x)
		x = layers.BatchNormalization()(x)

		x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

		# Project residual
		residual = layers.Conv2D(size, 1, strides=2, padding="same")(
			previous_block_activation
		)
		x = layers.add([x, residual])  # Add back residual
		previous_block_activation = x  # Set aside next residual

	x = layers.SeparableConv2D(1024, 3, padding="same")(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation("relu")(x)

	x = layers.GlobalAveragePooling2D()(x)
	if num_classes == 2:
		activation = "sigmoid"
		units = 1
	else:
		activation = "softmax"
		units = num_classes

	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(units, activation=activation)(x)
	
	return keras.Model(inputs, outputs)


#Script containing stuff from a tutorial
def preamble():
	print("Hello lol!")
	
	layer = keras.layers.Dense(3)
	layer.build((None, 4))  # Create the weights
	
	print("weights:", len(layer.weights))
	print("trainable_weights:", len(layer.trainable_weights))
	print("non_trainable_weights:", len(layer.non_trainable_weights))


	layer = keras.layers.BatchNormalization()
	layer.build((None, 4))  # Create the weights
	
	print("weights:", len(layer.weights))
	print("trainable_weights:", len(layer.trainable_weights))
	print("non_trainable_weights:", len(layer.non_trainable_weights))
	
	layer = keras.layers.Dense(3)
	layer.build((None, 4))  # Create the weights
	layer.trainable = False  # Freeze the layer
	
	print("weights:", len(layer.weights))
	print("trainable_weights:", len(layer.trainable_weights))
	print("non_trainable_weights:", len(layer.non_trainable_weights))
	
	
	print("\n\n\n\n\n#####################")
	
	# Make a model with 2 layers
	layer1 = keras.layers.Dense(3, activation="relu")
	layer2 = keras.layers.Dense(3, activation="sigmoid")
	model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])
	
	# Freeze the first layer
	layer1.trainable = False
	
	# Keep a copy of the weights of layer1 for later reference
	initial_layer1_weights_values = layer1.get_weights()
	
	# Train the model
	model.compile(optimizer="adam", loss="mse")
	model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
	
	# Check that the weights of layer1 have not changed during training
	final_layer1_weights_values = layer1.get_weights()
	np.testing.assert_allclose( 
		initial_layer1_weights_values[0], final_layer1_weights_values[0]
	)
	np.testing.assert_allclose(
		initial_layer1_weights_values[1], final_layer1_weights_values[1]
	)
	
	print("\n\n\n\n###############\n\n\n")
	print("Asserting that trainable status propagates recursively.")
	inner_model = keras.Sequential(
		[
			keras.Input(shape=(3,)),
			keras.layers.Dense(3, activation="relu"),
			keras.layers.Dense(3, activation="relu"),
		]
	)
	
	model = keras.Sequential(
		[keras.Input(shape=(3,)), inner_model, keras.layers.Dense(3, activation="sigmoid"),]
	)
	
	model.trainable = False  # Freeze the outer model
	
	assert inner_model.trainable == False  # All layers in `model` are now frozen
	assert inner_model.layers[0].trainable == False  # `trainable` is propagated recursively

	print("\n\n\n\n#########################################################\n\n\n")
	
	print("Now for the actual example using pretrained weights.\n\n")
	
	base_model = keras.applications.Xception(
		weights='imagenet',  # Load weights pre-trained on ImageNet.
		input_shape=(150, 150, 3),
		include_top=False)  # Do not include the ImageNet classifier at the top.

	base_model.trainable = False
	
	#New model to go on top of base_model
	inputs = keras.Input(shape=(150, 150, 3))
	# We make sure that the base_model is running in inference mode here,
	# by passing `training=False`. This is important for fine-tuning, as you will
	# learn in a few paragraphs.
	x = base_model(inputs, training=False)
	# Convert features of shape `base_model.output_shape[1:]` to vectors
	x = keras.layers.GlobalAveragePooling2D()(x)
	# A Dense classifier with a single unit (binary classification)
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)
	
	model.compile(
		optimizer=keras.optimizers.Adam(),
		loss=keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=[keras.metrics.BinaryAccuracy()])

#This part needs a new dataset loaded. I'll look at the rest of the tutorial first.
	# ~ model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
	
	print("\n\n\n\n#########################################################")
	print("Fine tuning...")

	# Unfreeze the base model
	base_model.trainable = True
	
	# It's important to recompile your model after you make any changes
	# to the `trainable` attribute of any inner layer, so that your changes
	# are take into account
	model.compile(
		optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
		loss=keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=[keras.metrics.BinaryAccuracy()])
	
	# Train end-to-end. Be careful to stop before you overfit!
	# ~ model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)
	
	#BIG NOTE: Whenever you change the trainable or other stuff in the
	#model, you must re-compile it otherwise it wont do much.

	print("\n\n\n\n#########################################################")
	print("If using a custom fit method instead of .fit()...")

	# Create base model
	base_model = keras.applications.Xception(
		weights='imagenet',
		input_shape=(150, 150, 3),
		include_top=False)
	# Freeze base model
	base_model.trainable = False
	
	# Create new model on top.
	inputs = keras.Input(shape=(150, 150, 3))
	x = base_model(inputs, training=False)
	x = keras.layers.GlobalAveragePooling2D()(x)
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)
	
	loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
	optimizer = keras.optimizers.Adam()
	
	# Iterate over the batches of a dataset.
	# ~ for inputs, targets in new_dataset:
		# ~ # Open a GradientTape.
		# ~ with tf.GradientTape() as tape:
			# ~ # Forward pass.
			# ~ predictions = model(inputs)
			# ~ # Compute the loss value for this batch.
			# ~ loss_value = loss_fn(targets, predictions)
	
	# Get gradients of loss wrt the *trainable* weights.
	# ~ gradients = tape.gradient(loss_value, model.trainable_weights)
	# ~ # Update the weights of the model.
	# ~ optimizer.apply_gradients(zip(gradients, model.trainable_weights))
	
	
	print("\n\n\n\n#########################################################")


#my evaluation grapher thingy. Saves graphs to file.
def performEvaluation(history, tmpFolder, model, test_ds):
	print("Performing evaluation...")
	
	scores = model.evaluate(test_ds)
	
	if IS_GLOBAL_PRINTING_ON:
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		print("history...")
		print(history)
		print("history.history...")
		print(history.history)
	
	accuracy = history.history["binary_accuracy"]
	val_accuracy = history.history["val_binary_accuracy"]
	
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	epochs = range(1, len(accuracy) + 1)
	plt.plot(epochs, accuracy, "o", label="Training accuracy")
	plt.plot(epochs, val_accuracy, "^", label="Validation accuracy")
	plt.title("Training and validation accuracy")
	plt.legend()
	plt.savefig(tmpFolder + "trainvalacc.png")
	plt.clf()
	
	plt.plot(epochs, loss, "o", label="Training loss")
	plt.plot(epochs, val_loss, "^", label="Validation loss")
	plt.title("Training and validation loss")
	plt.legend()
	plt.savefig(tmpFolder + "trainvalloss.png")
	plt.clf()


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
