from keras.models import *
from keras.layers import *

from .config import IMAGE_ORDERING
from model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder, ConvBlock, UpConvolution,Bottom_Layer
from .resnet50 import get_resnet50_encoder


if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1



def unet_mini( n_classes , input_height=360, input_width=480   ):

	if IMAGE_ORDERING == 'channels_first':
		img_input = Input(shape=(3,input_height,input_width))
	elif IMAGE_ORDERING == 'channels_last':
		img_input = Input(shape=(input_height,input_width , 3 ))


	conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(img_input)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)
	
	conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)
	
	conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv3)

	up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(conv3), conv2], axis=MERGE_AXIS)
	conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(up1)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv4)
	
	up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(conv4), conv1], axis=MERGE_AXIS)
	conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(up2)
	conv5 = Dropout(0.2)(conv5)
	conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING, activation='relu', padding='same')(conv5)
	
	o = Conv2D( n_classes, (1, 1) , data_format=IMAGE_ORDERING ,padding='same')(conv5)

	model = get_segmentation_model(img_input , o )
	model.model_name = "unet_mini"
	return model


def _unet( n_classes , encoder , l1_skip_conn=False,  input_height=416, input_width=608 ,filter_size=32 ):

    img_input , levels, last_node = encoder( input_height=input_height ,  input_width=input_width,filter_size=filter_size )
    print("Middle filters - {}".format(filter_size*16))
    bottom = Bottom_Layer(last_node, filter_size*16, 3, 1)
    up_input = bottom
    levels_len = len(levels)

    for level_i in range(1,levels_len + 1):
        up_input = UpConvolution(up_input,levels[levels_len - level_i],filter_size * (2**(levels_len - level_i)))
    output =  Conv2D( n_classes , (1, 1) , padding='same', data_format=IMAGE_ORDERING )( up_input )
    model = get_segmentation_model(img_input , output )
    return model


def unet(  n_classes ,  input_height=416, input_width=608 , encoder_level=3 ) : 
	
	model =  _unet( n_classes , vanilla_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "unet"
	return model


def vgg_unet( n_classes ,  input_height=416, input_width=608 , encoder_level=3):

	model =  _unet( n_classes , get_vgg_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "vgg_unet"
	return model


def resnet50_unet( n_classes ,  input_height=416, input_width=608 , encoder_level=3):

	model =  _unet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "resnet50_unet"
	return model



def mobilenet_unet( n_classes ,  input_height=224, input_width=224 , encoder_level=3):

	model =  _unet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_unet"
	return model

def f1(y_true, y_pred):
    '''
    Calculates the F1 by using keras.backend
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def UNet(filters_dims, activation='relu', kernel_initializer='glorot_uniform', padding='same'):
    inputs = Input((512, 512, 3))
    new_inputs = inputs
    conv_layers = []
    # Encoding Phase
    print("Encoding Phase")
    for i in range(len(filters_dims) - 1):
        print("Stage :", i+1)
        print("========================================")
        print(new_inputs.shape)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(new_inputs)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(conv)
        conv_layers.append(conv)
        new_inputs = MaxPooling2D(pool_size=(2, 2))(conv)
        print(new_inputs.shape)
        # op = BatchNormalization()(op)

    # middle phase
    print("middle phase")
    print("========================================")
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(new_inputs)
    conv = Conv2D(filters_dims[-1], 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_initializer)(conv)
    new_inputs = Dropout(0.5)(conv)
    print(new_inputs.shape)

    filters_dims.reverse()
    conv_layers.reverse()

    # Decoding Phase
    print("Decoding Phase")
    for i in range(1, len(filters_dims)):
        print(i)
        print("========================================")

        print(new_inputs.shape)
        up = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                    kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(new_inputs))
        concat = keras.layers.merge([conv_layers[i-1], up], mode='concat', concat_axis=3)
        conv = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                      kernel_initializer=kernel_initializer)(concat)
        new_inputs = Conv2D(filters_dims[i], 3, activation=activation, padding=padding,
                            kernel_initializer=kernel_initializer)(conv)
        print(new_inputs.shape)
    outputs = Conv2D(19, 1, activation='softmax', padding='same',
                     kernel_initializer='glorot_uniform')(new_inputs)
    print(outputs.shape)
    model = Model(input=inputs, output=outputs, name='UNet')
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', "categorical_accuracy", f1])
    return model

if __name__ == '__main__':
	m = unet_mini( 101 )
	m = _unet( 101 , vanilla_encoder )
	# m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
	m = _unet( 101 , get_vgg_encoder )
	m = _unet( 101 , get_resnet50_encoder )
	

