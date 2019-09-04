import sys
import skimage

from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_utils import getLogger

import tensorflow as tf
from pathlib import Path
import numpy as np

from ocrd_anybaseocr.mrcnn import model
from ocrd_anybaseocr.mrcnn import visualize
from ocrd_anybaseocr.mrcnn.config import Config

TOOL = 'ocrd-anybaseocr-block-segmentation'
LOG = getLogger('OcrdAnybaseocrBlockSegmenter')

class InferenceConfig(Config):
    NAME = "block"    
    IMAGES_PER_GPU = 1  
    NUM_CLASSES = 1 + 12      
    DETECTION_MIN_CONFIDENCE = 0.9

class OcrdAnybaseocrBlockSegmenter(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrBlockSegmenter, self).__init__(*args, **kwargs) 

    def process(self):
        
        if not tf.test.is_gpu_available():
            LOG.error("Your system has no CUDA installed. No GPU detected.")
            sys.exit(1)
        

        model_path = Path(self.parameter['block_segmentation_model'])
        model_weights = Path(self.parameter['block_segmentation_weights'])

        class_names = ['BG','page-number', 'paragraph', 'catch-word', 'heading', 'drop-capital', 'signature-mark','header',
                       'marginalia', 'footnote', 'footnote-continued', 'caption', 'endnote', 'footer','TOC-entry']

        
        if not Path(model_path).is_dir():
            LOG.error("""\
                Block Segmentation model was not found at '%s'. Make sure the `model_path` parameter
                points to the local model path.

                model can be downloaded from http://url
                """ % model_path)
            sys.exit(1)
        

            
        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            fname = pcgts.get_Page().imageFilename
            LOG.info("INPUT FILE %s", fname)
            config = InferenceConfig()            
            mrcnn_model = model.MaskRCNN(mode="inference", model_dir=str(model_weights), config=config)
            mrcnn_model.load_weights(str(model_weights), by_name=True)
            file_name=fname.split(".tif")[0]
            image = skimage.io.imread(fname, plugin='pil')
            results = mrcnn_model.detect([image], verbose=1)            
            r = results[0]        
            for class_id in r['class_ids']:                
                LOG.info("Block Class: %s", class_names[class_id])            
            LOG.info("ROIs: %s", np.array_str(r['rois']))
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,file_name, r['scores'])