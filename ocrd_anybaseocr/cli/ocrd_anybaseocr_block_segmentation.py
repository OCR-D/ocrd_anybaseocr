import sys
import skimage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #No prints from the tensorflow side

from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_utils import (
    getLogger, 
    concat_padded, 
    MIMETYPE_PAGE,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points
    )

import warnings
import ocrolib
warnings.filterwarnings('ignore',category=FutureWarning) 
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pathlib import Path
import numpy as np
import matplotlib.path as pltPath
from shapely.geometry import Polygon

from ocrd_anybaseocr.mrcnn import model
from ocrd_anybaseocr.mrcnn.config import Config
from ocrd_models.constants import NAMESPACES as NS

from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    AlternativeImageType,
    to_xml,
    MetadataItemType,
    LabelsType, LabelType,
    RegionRefIndexedType, OrderedGroupType, ReadingOrderType
)
from ocrd_models.ocrd_page_generateds import CoordsType



TOOL = 'ocrd-anybaseocr-block-segmentation'
LOG = getLogger('OcrdAnybaseocrBlockSegmenter')
FALLBACK_IMAGE_GRP = 'OCR-D-IMG-BLOCK-SEGMENT'

class InferenceConfig(Config):
    NAME = "block"    
    IMAGES_PER_GPU = 1  
    NUM_CLASSES = 1 + 17      
    DETECTION_MIN_CONFIDENCE = 0.9

class OcrdAnybaseocrBlockSegmenter(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrBlockSegmenter, self).__init__(*args, **kwargs) 
        #self.reading_order = []
        self.order = 0
        
    def process(self):
        
        if not tf.test.is_gpu_available():
            LOG.error("Your system has no CUDA installed. No GPU detected.")
            sys.exit(1)
        try:
            page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
            
        model_path = Path(self.parameter['block_segmentation_model'])
        
        model_weights = Path(self.parameter['block_segmentation_weights'])
        class_names = ['BG','page-number', 'paragraph', 'catch-word', 'heading', 'drop-capital', 'signature-mark','header',
                       'marginalia', 'footnote', 'footnote-continued', 'caption', 'endnote', 'footer','keynote',
                       'image','table', 'graphics']
        
        if not Path(model_weights).is_file():
            LOG.error("""\
                Block Segmentation model weights file was not found at '%s'. Make sure the `model_weights` parameter
                points to the local model weights path.
                """ % model_weights)
            sys.exit(1)
            
        config = InferenceConfig()
        mrcnn_model = model.MaskRCNN(mode="inference", model_dir=str(model_path), config=config)
        mrcnn_model.load_weights(str(model_weights), by_name=True)

        oplevel = self.parameter['operation_level']
        for (n, input_file) in enumerate(self.input_files):
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            page_id = input_file.pageId or input_file.ID 

            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_filter ='binarized,deskewed,cropped') 
            
            #Display Warning If image segment results already exist or not in StructMap?
            regions = page.get_TextRegion() + page.get_TableRegion()
            if regions:
                LOG.warning("Image already has text segments!")
            
            if oplevel=="page":
                self._process_segment(page_image, page, page_xywh, page_id, input_file, n, mrcnn_model, class_names)
            else:
                LOG.warning('Operation level %s, but should be "page".', oplevel)
                break
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)
            
            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            if file_id == input_file.ID:
                file_id = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts).encode('utf-8')
            )

    def _process_segment(self,page_image, page, page_xywh, page_id, input_file, n, mrcnn_model, class_names):
        #check for existing text regions and whether to overwrite them
        if page.get_TextRegion():
            if self.parameter['overwrite']:
                LOG.info('removing existing TextRegions in page "%s"', page_id)
                textregion.set_TextRegion([])
            else:
                LOG.warning('keeping existing TextRegions in page "%s"', page_id)
                return
        #check if border exists
        if page.get_Border():
            border_coords = page.get_Border().get_Coords()
            border_points = polygon_from_points(border_coords.get_points())
            
            border = Polygon(border_points)

        img_array = ocrolib.pil2array(page_image)
        if len(img_array.shape) <= 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        results = mrcnn_model.detect([img_array], verbose=1)    
        r = results[0]        
        
        
        # define reading order on basis of coordinates
        reading_order = []
        for i in range(len(r['rois'])):                
            
            width,height,_ = img_array.shape
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]
            if (min_x - 5) > width and r['class_ids'][i] == 2:
                min_x-=5
            if (max_x + 10) < width and r['class_ids'][i] == 2:
                min_x+=10
            reading_order.append((min_y, min_x))
            
        reading_order = sorted(reading_order, key=lambda reading_order:(reading_order[0], reading_order[1]))
        
        
        #Creating Reading Order object in PageXML
        order_group = OrderedGroupType(caption="Regions reading order",id=page_id)
        for i in range(len(r['rois'])):
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]
            if (min_x - 5) > width and r['class_ids'][i] == 2:
                min_x-=5
            if (max_x + 10) < width and r['class_ids'][i] == 2:
                min_x+=10
            region_polygon = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            
            if border:
                cut_region_polygon = border.intersection(Polygon(region_polygon))
                if cut_region_polygon.is_empty:
                    continue
            else:
                cut_region_polygon = Polygon(region_polygon)
                
            order_index = reading_order.index((min_y, min_x))
            region_id = '%s_region%04d' % (page_id, order_index)
            regionRefIndex = RegionRefIndexedType(index=order_index, regionRef=region_id)
            order_group.add_RegionRefIndexed(regionRefIndex)
            
        reading_order_object = ReadingOrderType()
        reading_order_object.set_OrderedGroup(order_group)
        page.set_ReadingOrder(reading_order_object)
        
        
        for i in range(len(r['rois'])):                
            width,height,_ = img_array.shape
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]
            
            if (min_x - 5) > width and r['class_ids'][i] == 2:
                min_x-=5
            if (max_x + 10) < width and r['class_ids'][i] == 2:
                min_x+=10
            region_polygon = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            
            cut_region_polygon = border.intersection(Polygon(region_polygon))
            if cut_region_polygon.is_empty:
                continue
            cut_region_polygon = [i for i in zip(list(cut_region_polygon.exterior.coords.xy[0]),list(cut_region_polygon.exterior.coords.xy[1]))][:-1]
            #print(cut_region_polygon)
            region_polygon = coordinates_for_segment(cut_region_polygon, page_image, page_xywh)
            region_points = points_from_polygon(region_polygon)
            
            read_order = reading_order.index((min_y, min_x))
            

            
            
            
            # this can be tested, provided whether we need previous comments or not?
            region_img = img_array[min_x:max_x,min_y:max_y] #extract from points and img_array
            region_img = ocrolib.array2pil(region_img)
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)
                
            file_path = self.workspace.save_image_file(region_img,
                                   file_id+"_"+str(i),
                                   page_id=page_id,
                                   file_grp=self.image_grp)
            
            # ai = AlternativeImageType(filename=file_path, comments=page_xywh['features'])
            region_id = '%s_region%04d' % (page_id, i)
            coords = CoordsType(region_points)
            
            #incase of imageRegion
            if r['class_ids'][i] == 15:
                image_region = ImageRegionType(custom='readingOrder {index:'+str(read_order)+';}',id=region_id ,Coords=coords, type_=class_names[r['class_ids'][i]])
                # image_region.add_AlternativeImage(ai)
                page.add_ImageRegion(image_region)
                continue
            if r['class_ids'][i] == 16:
                table_region = TableRegionType(custom='readingOrder {index:'+str(read_order)+';}',id=region_id ,Coords=coords, type_=class_names[r['class_ids'][i]])
                # table_region.add_AlternativeImage(ai)
                page.add_TableRegion(table_region)
                continue
            if r['class_ids'][i] == 17:
                graphic_region = GraphicRegionType(custom='readingOrder {index:'+str(read_order)+';}',id=region_id ,Coords=coords, type_=class_names[r['class_ids'][i]])
                # graphic_region.add_AlternativeImage(ai)
                page.add_GraphicRegion(graphic_region)
                continue
            
            textregion = TextRegionType(custom='readingOrder {index:'+str(read_order)+';}',id=region_id ,Coords=coords, type_=class_names[r['class_ids'][i]])
            # textregion.add_AlternativeImage(ai)
            
            #border = page.get_Border()
            #if border:
            #    border.add_TextRegion(textregion)
            #else:
            page.add_TextRegion(textregion)