import os
import sys
import pickle
import argparse
import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.layers import Input
from keras.models import load_model
#from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score
from keras.applications import inception_v3, resnet50
from keras.preprocessing.image import ImageDataGenerator

from ..constants import OCRD_TOOL

from ocrd import Processor
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class OcrdAnybaseocrLayoutAnalyser(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-layout-analysis']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrLayoutAnalyser, self).__init__(*args, **kwargs)

    def define_model(model = 'inception_v3', num_classes=34, input_size=(600, 500, 1)):
        input_dims = Input(shape=(input_size))
        if model == "inception_v3":
            model = inception_v3.InceptionV3(include_top=True, weights=None, classes=classes, input_tensor=input_dims)
        elif model == "resnet50":
            model = resnet50.ResNet50(include_top=True, weights=None, input_tensor=input_dims, classes=classes)
        else:
            print('wrong input')
            sys.exit(0)                      
                
    def create_model(path, model_name='inception_v3', def_weights=True, num_classes=34, input_size=(600, 500, 1)):
        ''' 
            num_classes : number of classes to predict for
            input_size  : size of input dimension of image 
        '''
        if def_weights:        
            model = load_model(path)
        else:
            model = define_model(model_name, num_classes, input_size)
            model.load_weights(path)                            
        return model

    def start_test(model, img_array, filename, labels):
        # shape should be 600,500, 1
        pred = model.predict(img_array)
        pred_classes = np.argmax(pred, axis=1)

        # convert label
        # labels = train_generator.class_indices
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in pred_classes]            
        return predictions[0]
        

    def process(self):    	
        if not tf.test.is_gpu_available():
            print("Your system has no CUDA installed. No GPU detected.")
            sys.exit(1)

        path = Path(self.parameter['model_path']).absolute()

        if not Path(path).is_dir():
            print("""\
                Layout Classfication model was not found at '%s'. Make sure the `model_path` parameter
                points to the local model path.

                model can be downloaded from http://url
                """ % path)
            sys.exit(1)
        else:
            print('Loading model from file ', path)
            model = create_model(path)    
            # load the mapping
            pickle_in = open("mapping_"+path.split('/')[1][:-5]+".pickle", "rb")
            class_indices = pickle.load(pickle_in)

        for (_, input_file) in enumerate(self.input_files):
            local_input_file = self.workspace.download_file(input_file)
            pcgts = parse(local_input_file.url, silence=True)            
            fname = pcgts.get_Page().imageFilename
            img = self.workspace.resolve_image_as_pil(fname)
            img_array = ocrolib.pil2array(img)            
            results = start_test(model, img_array, fname, class_indices)