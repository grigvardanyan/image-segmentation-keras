
import os
import cv2
import glob
import ntpath
import random
import keras
import itertools
import numpy as np
from tqdm import tqdm
from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg

random.seed(0)
class_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(5000)]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_path, y_path, batch_size=4, dim=(512,512,3), n_channels=3,
                 n_classes=19 ,shuffle=True):
        'Initialization'
        img_id_seg_pairs , images = get_pairs_from_paths( x_path , y_path )
        self.images = images
        self.dim = dim
        self.batch_size = batch_size
        self.labels = img_id_seg_pairs
        self.list_IDs = list(img_id_seg_pairs.keys())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
     def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = []
        Y = []

        # Generate data
        for key in list_IDs_temp:
            try:
                im , seg = get_path(self.images , int(key)) , self.labels[key]
            except Exception as ex:
                print(key)
                input("Error occured")
            im = cv2.imread(im , 1 )
            
            X.append( get_image_arr(im , 512 , 512 ,odering=IMAGE_ORDERING )  )
            Y.append( get_segmentation_arr( seg , self.n_classes , 512 , 512 )  )

        return np.array(X) , np.array(Y)
    
    
def get_pairs_from_paths( images_path , segs_path):
    images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg") )
    segmentations = glob.glob( os.path.join(segs_path,"*.png") )
    input_output = dict()
    n_classes = 19
    lables = ["background","skin","nose","eye_g","l_eye","r_eye","l_brow","r_brow","l_ear","r_ear","mouth","u_lip","l_lip","hair","hat","ear_r","neck_l","neck","cloth"]

    for seg_img in segmentations:
        parts = seg_img.split("/")
        id = parts[len(parts) - 1 ].split("_")[0]
        if parts[len(parts) -1 ] == id :
            assert((parts[len(parts) - 1 ] == id), ("Current file doesn't has proper file format it need to has following format index_mask: " + seg_img))
        if id not in input_output:
            input_output[id] = [""] * n_classes

        for i in range(len(lables)):
            if lables[i] in seg_img:
                input_output[id][i] = seg_img
                break
    return input_output, images

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
    #if no_reshape:
        #return seg_labels

    #seg_labels = np.reshape(seg_labels, ( width * height , nClasses ))
    return seg_labels

def verify_segmentation_dataset( images_path , segs_path , n_classes ):
    img_seg_pairs, images = get_pairs_from_paths( images_path , segs_path )
    assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "

    for id in tqdm(img_seg_pairs) :
        img_path = get_path(images,int(id))
        img = cv2.imread( img_path )
        for seg_path in img_seg_pairs[id]:
            seg = cv2.imread( seg_path )
            assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ img_path + " " + seg_path
    print("Dataset verified! ")

def image_segmentation_generator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False ):
    img_id_seg_pairs , images = get_pairs_from_paths( images_path , segs_path )
    keys = list(img_id_seg_pairs.keys())
    random.shuffle(keys)
    random.shuffle(keys)
    random.shuffle(keys)
    img_id_seg_pairs = dict([(key,img_id_seg_pairs[key])for key in keys])
    zipped = itertools.cycle( img_id_seg_pairs  )

    while True:
        X = []
        Y = []

        for _ in range( batch_size) :
            key = next(zipped)
            try:
                im , seg = get_path(images , int(key)) , img_id_seg_pairs[key]
            except Exception as ex:
                print(key)
                input("Error occured")
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
