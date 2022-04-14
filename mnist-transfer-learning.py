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
	
	
	

	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
