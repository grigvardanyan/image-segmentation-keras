from keras.models import *
from keras.layers import *

import keras.backend as K
from keras_segmentation.models.config import IMAGE_ORDERING

if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1

def ConvBlock(input ,
              filter_size,
              pad = 1,
              kernel = 3,
              pool_size = 2,
              pooling = False,
              droprate = 0.25,
              dropout = False
              ):
    x = (ZeroPadding2D((pad,pad) , data_format=IMAGE_ORDERING ))( input )
    x = (Conv2D(filter_size, (kernel, kernel) , data_format=IMAGE_ORDERING , padding='valid'))( x )
    x = (BatchNormalization())( x )
    x = (Activation('relu'))( x )
    pooled = x	
    if pooling:
	    pooled = MaxPooling2D((pool_size, pool_size),data_format=IMAGE_ORDERING)(x)
    if dropout:
	    pooled = Dropout(droprate)(pooled)
    if pooling:
	    return x,pooled
    return x

def UpConvolution(input,
                  left_conv,
                  filter_size
                  ):
    conv_transpose =UpSampling2D(size=(2,2))(input)# Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(input)
    merge = ( concatenate([ conv_transpose ,left_conv],axis=MERGE_AXIS )  )
    batch_norm = BatchNormalization()(merge)
    conv = ConvBlock(batch_norm, filter_size)
    conv = ConvBlock(conv, filter_size)
    print("Upconv filters - {}".format(filter_size))
    return conv

def Bottom_Layer(input,
                 filter_size,
                 kernel,
                 pad):
    bottom_1 = ConvBlock(input,filter_size)
    bottom_2 = ConvBlock(bottom_1,filter_size,droprate=0.5,dropout=True )
    return bottom_2

def vanilla_encoder( input_height=224 ,
                  input_width=224,
                  kernel = 3,
                  filter_size = 32,
                  pad = 1,
                  pool_size = 2 
                 ):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3,input_height,input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height,input_width , 3 ))
    levels = []

    conv = ConvBlock(img_input , filter_size)
    conv , conv_pooled  = ConvBlock(conv ,filter_size, pooling=True)
    levels.append(conv)

    for i in range(1,4):
	print("Down conv filters- {}".format((2**i) * filter_size))
        conv = ConvBlock(conv_pooled , (2**i) * filter_size)
        conv , conv_pooled = ConvBlock(conv , (2**i) * filter_size, pooling=True)
        levels.append(conv)
    return img_input, levels , conv_pooled

