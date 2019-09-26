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



def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

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
	
	#K.set_floatx('float16')
	
	if  isinstance(model, six.string_types) : # check if user gives model name insteead of the model object
		# create the model from the name
		assert ( not n_classes is None ) , "Please provide the n_classes"
		#if (not input_height is None ) and ( not input_width is None):
		#	model = model_from_name[ model ](  n_classes , input_height=input_height , input_width=input_width )
		#else:
		#	model = model_from_name[ model ](  n_classes )

	#n_classes = model.n_classes
	#input_height = model.input_height
	#input_width = model.input_width
	#output_height = model.output_height
	#output_width = model.output_width


	if validate:
		assert not (  val_images is None ) 
		assert not (  val_annotations is None ) 

	#if not optimizer_name is None:
	#	model.compile(loss='categorical_crossentropy',
	#		optimizer= optimizer_name ,
	#		metrics=['accuracy','categorical_accuracy'])
	
	#if not checkpoints_path is None:
	#	open( checkpoints_path+"_config.json" , "w" ).write( json.dumps( {
	#		"model_class" : model.model_name ,
	#		"n_classes" : n_classes ,
	#		"input_height" : input_height ,
	#		"input_width" : input_width ,
	#		"output_height" : output_height ,
	#		"output_width" : output_width 
	#	}))
	model = build_res_unet((512,512,3))#UNet([64, 128, 256, 512])    
	model.compile(loss='categorical_crossentropy',optimizer= optimizer_name ,metrics=['accuracy',,'categorical_accuracy'])

	if ( not (load_weights is None )) and  len( load_weights ) > 0:
		print("Loading weights from " , load_weights )
		model.load_weights(load_weights)

	if auto_resume_checkpoint and ( not checkpoints_path is None ):
		latest_checkpoint = find_latest_checkpoint( checkpoints_path )
		if not latest_checkpoint is None:
			print("Loading the weights from latest checkpoint "  ,latest_checkpoint )
			model.load_weights( latest_checkpoint )

	verify_dataset = False
	if verify_dataset:
		print("Verifying train dataset")
		verify_segmentation_dataset( train_images , train_annotations , n_classes )
		if validate:
			print("Verifying val dataset")
			verify_segmentation_dataset( val_images , val_annotations , n_classes )

	train_gen = DataGenerator(train_images,train_annotations,batch_size)
	#image_segmentation_generator( train_images , train_annotations ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   )
	validate = True
	print("Validate ************************************************************************")
	print(validate)
	print("Float type***********************************************************************")
	print(K.floatx())
	model_path = "/home/ubuntu/grig/cp-{epoch:02d}-{loss:.2f}.hdf5"
	cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path,verbose=1,period=2)
	
	if validate:
		print("Generate Validate*************************************")
	val_gen  = DataGenerator(val_images,val_annotations,val_batch_size)
		#image_segmentation_generator( val_images , val_annotations ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
	if not validate:
		for ep in range( epochs ):
			print("Starting Epoch" , ep )
			model.fit_generator( train_gen , steps_per_epoch  , epochs=1 ,use_multiprocessing=False)
			if not checkpoints_path is None:
				model.save_weights( checkpoints_path + "." + str( ep ) )
				print("saved " , checkpoints_path + ".model." + str( ep ) )
			print("Finished Epoch" , ep )
	else:
		print("With Validate")
		#for ep in range( epochs ):
		#print("Starting Epoch with Validate" , ep)
		history = model.fit_generator( train_gen  , validation_data=val_gen ,steps_per_epoch=steps_per_epoch,validation_steps=2399, epochs=epochs ,use_multiprocessing=False,callbacks=[cp_callback])
		with open(checkpoints_path, 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
		if not checkpoints_path is None:
			model.save_weights( checkpoints_path + "." + str( ep )  )
			#print("saved " , checkpoints_path + ".model." + str( ep ) )
			#print("Finished Epoch" , ep )





