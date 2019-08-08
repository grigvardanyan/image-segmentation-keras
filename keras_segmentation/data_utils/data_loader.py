
import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
import ntpath

from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg
import random

random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]

def get_pairs_from_paths( images_path , segs_path ):
	images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg")  )
	segmentations  =  glob.glob( os.path.join(segs_path,"*.png")  ) 

    input_output = dict()
    n_classes = 18
    lables = ["background","skin","nose","eye_g","l_eye","r_eye","l_brow","r_brow","l_ear","r_ear","mouth","u_lip","l_lip","hair","hat","ear_r","neck_l","neck","cloth"]

    for seg_img in segmentations:
        id = seg_img.split("_")[0]
        assert(seg_img == id), ("Current file doesn't has proper file format it need to has following format index_mask: " + seg_img)

        id = int(id)
        if id not in input_output:
            input_output[id] = [""] * n_classes

        for i in range(len(lables)):
            if lables[i] in seg_img:
                input_output[id][i] = seg_img
	return input_output,images

def get_image_arr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):


	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	if imgNorm == "sub_and_divide":
		img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
	elif imgNorm == "sub_mean":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img[:,:,0] -= 103.939
		img[:,:,1] -= 116.779
		img[:,:,2] -= 123.68
		img = img[ : , : , ::-1 ]
	elif imgNorm == "divide":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img = img/255.0

	if odering == 'channels_first':
		img = np.rollaxis(img, 2, 0)
	return img

def get_segmentation_arr( paths , nClasses ,  width , height , no_reshape=False ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	for i in range(len(paths)):
        if paths[i] != "":
	        img = cv2.imread(paths[i], 0)
            img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
            seg_labels[:,:,i] = (img == 255 ).astype(int)
        
	
	if no_reshape:
		return seg_labels

	seg_labels = np.reshape(seg_labels, ( width * height , nClasses ))
	return seg_labels

def verify_segmentation_dataset( images_path , segs_path , n_classes ):
	
	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "
	
	for im_fn , seg_fn in tqdm(img_seg_pairs) :
		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )

		assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
		assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

	print("Dataset verified! ")


def image_segmentation_generator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False ):
	

	img_id_seg_pairs , images = get_pairs_from_paths( images_path , segs_path )
	random.shuffle( img_seg_pairs )
	zipped = itertools.cycle( img_id_seg_pairs  )

	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
            key = next(zipped)
			im , seg = get_path(images , key) , img_id_seg_pairs[key]

			im = cv2.imread(im , 1 )

			if do_augment:
				img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

			X.append( get_image_arr(im , input_width , input_height ,odering=IMAGE_ORDERING )  )
			Y.append( get_segmentation_arr( seg , n_classes , output_width , output_height )  )

		yield np.array(X) , np.array(Y)

def get_path(images,id):
    for img in images:
        img_name = ntpath.basename(img).split('.')[0]
        if len(img_name) == get_length(id) and int(img_name) == id:
            return img

def get_length(id):
    length = 0
    while True:
        id = id//10
        length += 1
        if id == 0:
            break
    return length