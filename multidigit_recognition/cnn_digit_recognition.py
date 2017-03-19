# Import useful functions from keras library
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def cnn_digit(input_shape=(64, 64, 1), localization=False, add_dropout=True, prob_dropout=0.5):

	# Input data
	main_input = Input(shape=input_shape, name='main_input')

	# Main network
	x = Convolution2D(48, 5, 5, init='glorot_uniform', border_mode='same', activation='relu')(main_input)
	x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
	x = BatchNormalization()(x)
	x = Convolution2D(64, 5, 5, init='glorot_uniform', border_mode='same', subsample=(1, 1),
			      activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
	x = BatchNormalization()(x)
	x = Convolution2D(128, 5, 5, init='glorot_uniform', border_mode='same', subsample=(1, 1),
			      activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
	x = BatchNormalization()(x)
	x = Convolution2D(160, 5, 5, init='glorot_uniform', border_mode='same', subsample=(1, 1),
			      activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=(1,1))(x)
	x = BatchNormalization()(x)
	x = Convolution2D(192, 5, 5, init='glorot_uniform', border_mode='same', subsample=(1, 1),
			      activation='relu')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
	x = BatchNormalization()(x)
	x = Flatten()(x)
	x = Dense(1024, activation='relu')(x)

	# Add dropout if required
	if add_dropout:
		x = Dropout(prob_dropout)(x)

	# Classification fully connected layers
	y = Dense(1024, activation='relu')(x)
	y = Dense(524, activation='relu')(y)

	# Five separate classifiers
	first_output = Dense(11, activation='softmax', name='first_softmax')(y)
	second_output = Dense(11, activation='softmax', name='second_softmax')(y)
	third_output = Dense(11, activation='softmax', name='third_softmax')(y)
	fourth_output = Dense(11, activation='softmax', name='fourth_softmax')(y)
	fifth_output = Dense(11, activation='softmax', name='fifth_softmax')(y)

	if localization:
		# Regression fully connected layers
		z = Dense(1024, activation='relu')(x)
		z = Dense(524, activation='relu')(z)

		# One regression head
		sixth_output = Dense(20, activation=None, name='regression_head')(z)

		model = Model(input=[main_input], output=[first_output, second_output,
					third_output, fourth_output, fifth_output, sixth_output])

	else:
		model = Model(input=[main_input], output=[first_output, second_output,
					third_output, fourth_output, fifth_output])

	return model
