#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  mnist-transfer-learning.py
#  
#  2022 Stephen Stengel <stephen.stengel@cwu.edu>

#A test program to see how transfer learning works.
#Tutorial sources:
# ~ https://keras.io/guides/transfer_learning/
# ~ https://towardsdatascience.com/transfer-learning-using-pre-trained-alexnet-model-and-fashion-mnist-43898c2966fb?gi=1f9cc1728578
# ~ https://www.kaggle.com/code/muerbingsha/mnist-vgg19/notebook


print("Running imports...")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist		#images of digits.
from keras.datasets import fashion_mnist  #images of clothes
from keras.datasets import cifar10    #small images

print("Done!")

def main(args):
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



	return 0


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
