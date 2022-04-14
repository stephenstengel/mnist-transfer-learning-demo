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


print("Running imports...")

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist		#images of digits.
from keras.datasets import fashion_mnist  #images of clothes
from keras.datasets import cifar10    #small images
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Done!")

def main(args):
	# ~ preamble()
	
	print("heh")
	xceptCatDog()

	return 0

def xceptCatDog():
	#Starting by importing the dataset manually because the first tutorial just doesn't compile.
	#Using this other tutorial: https://keras.io/examples/vision/image_classification_from_scratch/
	
	print("Cleaning out images that the tutorial doesn't like.")
	num_skipped = 0
	for folder_name in ("Cat", "Dog"):
		folder_path = os.path.join("PetImages", folder_name)
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
		"PetImages",
		validation_split=0.2,
		subset="training",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		"PetImages",
		validation_split=0.2,
		subset="validation",
		seed=1337,
		image_size=image_size,
		batch_size=batch_size,
	)
	
	print("Done!")
	
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
	train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
	validation_ds = val_ds.map(lambda x, y: (tf.image.resize(x, size), y)) #RENAME val_ds

	####### This makes the shapes wonky and causes crash.
	# ~ batch_size = 32
	# ~ train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
	# ~ validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
	
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
	
	epochs = 20
	print("train_ds: " + str(train_ds))
	# ~ print("train_ds shape: " + str(train_ds.shape))
	model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
	
	print("Done!")
	


#an xception model.
def make_model(input_shape, num_classes):
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


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
