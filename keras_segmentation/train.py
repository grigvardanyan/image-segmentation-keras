import argparse
import json
from .data_utils.data_loader import image_segmentation_generator , verify_segmentation_dataset,DataGenerator
from .models import model_from_name
import os
import six
from keras import backend as K
import tensorflow as tf
from .models.unet import UNet
from .models.resnet50 import build_res_unet,res_net
import pickle

def find_latest_checkpoint( checkpoints_path ):
	ep = 0
	r = None
	while True:
		if os.path.isfile( checkpoints_path + "." + str( ep )  ):
			r = checkpoints_path + "." + str( ep ) 
		else:
			return r 

		ep += 1

def iou_coef(y_true, y_pred):
    smooth=1
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    return (intersection + smooth) / ( union + smooth)

def mean_iou(y_true, y_pred):
	acc = 0
	for i in range(0,19):
		acc+=iou_coef(y_true[:,:,i],y_pred[:,:,i])
	acc = acc/19
	return acc

def dice_coef(y_true, y_pred):
	smooth=1e-7
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def mean_dice_coef(y_true, y_pred):
	acc = 0
	for i in range(0,19):
		acc+=dice_coef(y_true[:,:,i],y_pred[:,:,i])
	acc = acc/19
	return acc


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky_loss(y_true, y_pred):
	beta = 0.7
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	numerator = K.sum(y_true_f * y_pred_f)
	denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
	return 1 - (numerator + 1) / (K.sum(denominator) + 1)


def train( model  , 
		train_images  , 
		train_annotations , 
		input_height=None , 
		input_width=None , 
		n_classes=None,
		verify_dataset=False,
		checkpoints_path=None , 
		epochs = 5,
		batch_size = 2,
		validate=False , 
		val_images=None , 
		val_annotations=None ,
		val_batch_size=2 , 
		auto_resume_checkpoint=False ,
		load_weights=None ,
		steps_per_epoch=512,
		optimizer_name='adadelta' 
	):
	

	if validate:
		assert not (  val_images is None ) 
		assert not (  val_annotations is None ) 

	model = res_net((512,512,3))   
	model.compile(loss=tversky_loss,optimizer= optimizer_name ,metrics=['accuracy',mean_iou, mean_dice_coef])

	if ( not (load_weights is None )) and  len( load_weights ) > 0:
		print("Loading weights from " , load_weights )
		model.load_weights(load_weights)

	if auto_resume_checkpoint and ( not checkpoints_path is None ):
		latest_checkpoint = find_latest_checkpoint( checkpoints_path )
		if not latest_checkpoint is None:
			print("Loading the weights from latest checkpoint "  ,latest_checkpoint )
			model.load_weights( latest_checkpoint )

	#Verify dataset
	validate = True
	verify_dataset = False
	if verify_dataset:
		print("Verifying train dataset")
		verify_segmentation_dataset( train_images , train_annotations , n_classes )
		if validate:
			print("Verifying val dataset")
			verify_segmentation_dataset( val_images , val_annotations , n_classes )
			
    #Data Generatore
	train_gen = DataGenerator(train_images,train_annotations,batch_size)
	val_gen  = DataGenerator(val_images,val_annotations,val_batch_size)
	#Saving during training
	model_path = "/home/ubuntu/grig/cp-{epoch:02d}-{loss:.2f}.hdf5"
	cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path,verbose=1,period=2)
	
	#Train
	history = model.fit_generator( train_gen  , validation_data=val_gen ,epochs=epochs ,use_multiprocessing=False,callbacks=[cp_callback])
	with open(checkpoints_path, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	if not checkpoints_path is None:
		model.save_weights( checkpoints_path + "." + str( 60 )  )
			





