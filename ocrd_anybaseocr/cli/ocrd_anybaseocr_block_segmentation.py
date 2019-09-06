import os
import sys
import random
import math
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt

from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml, parse, TextRegionType
from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE
from ocrd_models.ocrd_page_generateds import RegionType

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) 
import tensorflow as tf
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from pathlib import Path
import ocrolib
from PIL import Image



# Root directory of the project
ROOT_DIR = os.path.abspath("/b_test/bymana/ocrd_demo/block_segmentation")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "samples/blocks/logs/good_model_old_data/")
#DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "samples/blocks/logs")

class InferenceConfig(Config):
    NAME = "block"    
    IMAGES_PER_GPU = 1  
    NUM_CLASSES = 1 + 12  
    STEPS_PER_EPOCH = 1000      
    DETECTION_MIN_CONFIDENCE = 0.9


class OcrdAnybaseocrBlockSegmenter(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-block-segmentation']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrBlockSegmenter, self).__init__(*args, **kwargs)	

    def process(self):

        
        if not tf.test.is_gpu_available():
            print("Your system has no CUDA installed. No GPU detected.")
            sys.exit(1)
        

        model_path = Path(self.parameter['block_segmentation_model'])
        model_weights = Path(self.parameter['block_segmentation_weights'])

        class_names = ['BG','page-number', 'paragraph', 'catch-word', 'heading', 'drop-capital', 'signature-mark','header',
                       'marginalia', 'footnote', 'footnote-continued', 'caption', 'endnote', 'footer','TOC-entry']

        
        if not Path(model_path).is_dir():
            print("""\
                Block Segmentation model was not found at '%s'. Make sure the `model_path` parameter
                points to the local model path.

                model can be downloaded from http://url
                """ % model_path)
            sys.exit(1)
        

            
        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            fname = pcgts.get_Page().imageFilename
            print(fname)            
            config = InferenceConfig()
            print(model_path, model_weights)
            model = modellib.MaskRCNN(mode="inference", model_dir=str(model_path), config=config)    		
            model.load_weights(str(model_weights), by_name=True)
            file_name=fname.split(".tif")[0]
            print(file_name)
            image = skimage.io.imread(fname, plugin='pil')
            results = model.detect([image], verbose=1)            
            r = results[0]
            for class_id in r['class_ids']:
                print("Block Class : ",class_names[class_id])
            print("ROIs : ",r['rois'])
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,file_name, r['scores'])
            
    		
