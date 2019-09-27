import argparse
import json
from .data_utils.data_loader import image_segmentation_generator , verify_segmentation_dataset,DataGenerator
from .models import model_from_name
import os
import six
from keras import backend as K
import tensorflow as tf
from .models.unet import UNet
from .models.resnet50 import build_res_unet
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



def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
    return (tf.math.log1p(tf.math.exp(-tf.math.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

def loss(y_true, y_pred):
	alpha = 0.25
	gamma = 2
	y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
	logits = tf.math.log(y_pred / (1 - y_pred))
	
	loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    # or reduce_sum and/or axis=-1
	return tf.reduce_mean(loss)

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

	model = build_res_unet((512,512,3))   
	model.compile(loss=loss,optimizer= optimizer_name ,metrics=['accuracy'])

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
	#image_segmentation_generator( train_images , train_annotations ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   )
	
	val_gen  = DataGenerator(val_images,val_annotations,val_batch_size)
	#image_segmentation_generator( val_images , val_annotations ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

	#Saving during training
	model_path = "/home/ubuntu/grig/cp-{epoch:02d}-{loss:.2f}.hdf5"
	cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path,verbose=1,period=2)
	
	#Train
	history = model.fit_generator( train_gen  , validation_data=val_gen ,steps_per_epoch=steps_per_epoch,validation_steps=2399, epochs=epochs ,use_multiprocessing=False,callbacks=[cp_callback])
	with open(checkpoints_path, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	if not checkpoints_path is None:
		model.save_weights( checkpoints_path + "." + str( 60 )  )
			





