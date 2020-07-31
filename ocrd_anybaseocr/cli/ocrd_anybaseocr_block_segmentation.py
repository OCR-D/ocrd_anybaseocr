# pylint: missing-module-docstring, missing-class-docstring, invalid-name
# pylint: disable=line-too-long, import-error, no-name-in-module, too-many-statements
# pylint: disable=wrong-import-position, wrong-import-order, too-many-locals, too-few-public-methods
import sys
import os
from pathlib import Path
import warnings
import click

import cv2
import numpy as np
from shapely.geometry import Polygon
import ocrolib


from ocrd import Processor
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_modelfactory import page_from_file
from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE,
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points
)
from ocrd_models.ocrd_page import (
    CoordsType,
    TextRegionType,
    GraphicRegionType,
    TableRegionType,
    ImageRegionType,
    to_xml,
    MetadataItemType,
    LabelsType, LabelType,
    RegionRefIndexedType, OrderedGroupType, ReadingOrderType
)
from ..mrcnn import model
from ..mrcnn.config import Config
from ..constants import OCRD_TOOL
from ..tensorflow_importer import tf


TOOL = 'ocrd-anybaseocr-block-segmentation'
LOG = getLogger('OcrdAnybaseocrBlockSegmenter')
FALLBACK_IMAGE_GRP = 'OCR-D-IMG-BLOCK-SEGMENT'


class InferenceConfig(Config):

    def __init__(self, confidence):
        Config.__init__(self, confidence)

    NAME = "block"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 14

#     NAME = "block"
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 14
#     DETECTION_MIN_CONFIDENCE = 0.9 # needs to be changed back to parameter
    #     DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE #taken as a parameter from tools.json


class OcrdAnybaseocrBlockSegmenter(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrBlockSegmenter, self).__init__(*args, **kwargs)
        #self.reading_order = []
        self.order = 0

    def process(self):

        if not tf.test.is_gpu_available():
            LOG.warning("Tensorflow cannot detect CUDA installation. Running without GPU will be slow.")
        try:
            page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)

        model_path = Path(self.parameter['block_segmentation_model'])
        model_weights = Path(self.parameter['block_segmentation_weights'])

        confidence = self.parameter['DETECTION_MIN_CONFIDENCE']
#         DETECTION_MIN_CONFIDENCE = Path(self.parameter['DETECTION_MIN_CONFIDENCE'])

        class_names = ['BG', 'page-number', 'paragraph', 'catch-word', 'heading', 'drop-capital', 'signature-mark', 'header',
                       'marginalia', 'footnote', 'footnote-continued', 'caption', 'endnote', 'footer', 'keynote',
                       'image', 'table', 'graphics']

        if not Path(model_weights).is_file():
            LOG.error("""\
                Block Segmentation model weights file was not found at '%s'. Make sure the `model_weights` parameter
                points to the local model weights path.
                """, model_weights)
            sys.exit(1)

#         config = InferenceConfig(Config,DETECTION_MIN_CONFIDENCE)

        config = InferenceConfig(confidence)
#         config = InferenceConfig()
        mrcnn_model = model.MaskRCNN(mode="inference", model_dir=str(model_path), config=config)
        mrcnn_model.load_weights(str(model_weights), by_name=True)

        oplevel = self.parameter['operation_level']
        for (n, input_file) in enumerate(self.input_files):

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            page_id = input_file.pageId or input_file.ID

            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(  # externalRef="parameter",
                                         Label=[LabelType(type_=name,
                                                          value=self.parameter[name])
                                                for name in self.parameter.keys()])]))

            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_filter='binarized,deskewed,cropped,clipped,non_text')
            # try to load pixel masks
            try:
                mask_image, mask_xywh, mask_image_info = self.workspace.image_from_page(page, page_id, feature_selector='clipped', feature_filter='binarized,deskewed,cropped,non_text')
            except:
                mask_image = None
            # Display Warning If image segment results already exist or not in StructMap?
            regions = page.get_TextRegion() + page.get_TableRegion()
            if regions:
                LOG.warning("Image already has text segments!")

            if oplevel == "page":
                self._process_segment(page_image, page, page_xywh, page_id, input_file, n, mrcnn_model, class_names, mask_image)
            else:
                LOG.warning('Operation level %s, but should be "page".', oplevel)
                break
            file_id = input_file.ID.replace(self.input_file_grp, page_grp)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            if file_id == input_file.ID:
                file_id = concat_padded(page_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts).encode('utf-8')
            )

    def _process_segment(self, page_image, page, page_xywh, page_id, input_file, n, mrcnn_model, class_names, mask):
        # check for existing text regions and whether to overwrite them
        border = None
        if page.get_TextRegion():
            if self.parameter['overwrite']:
                LOG.info('removing existing TextRegions in page "%s"', page_id)
                textregion.set_TextRegion([])
            else:
                LOG.warning('keeping existing TextRegions in page "%s"', page_id)
                return
        # check if border exists
        if page.get_Border():
            border_coords = page.get_Border().get_Coords()
            border_points = polygon_from_points(border_coords.get_points())
            border = Polygon(border_points)
#            page_image, page_xy = self.workspace.image_from_segment(page.get_Border(), page_image, page_xywh)

        img_array = ocrolib.pil2array(page_image)
        page_image.save('./checkthis.png')
        if len(img_array.shape) <= 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        results = mrcnn_model.detect([img_array], verbose=1)
        r = results[0]

        th = self.parameter['th']
        # check for existing semgentation mask
        # this code executes only when use_deeplr is set to True in ocrd-tool.json file
        if mask:
            mask = ocrolib.pil2array(mask)
            mask = mask//255
            mask = 1-mask
            # multiply all the bounding box part with 2
            for i in range(len(r['rois'])):

                min_x = r['rois'][i][0]
                min_y = r['rois'][i][1]
                max_x = r['rois'][i][2]
                max_y = r['rois'][i][3]
                mask[min_x:max_x, min_y:max_y] *= i+2
            cv2.imwrite('mask_check.png', mask*(255/(len(r['rois'])+2)))

            # check for left over pixels and add them to the bounding boxes
            pixel_added = True

            while pixel_added:

                pixel_added = False
                left_over = np.where(mask == 1)
                for x, y in zip(left_over[0], left_over[1]):
                    local_mask = mask[x-th:x+th, y-th:y+th]
                    candidates = np.where(local_mask > 1)
                    candidates = [k for k in zip(candidates[0], candidates[1])]
                    if len(candidates) > 0:
                        pixel_added = True
                        # find closest pixel with x>1
                        candidates.sort(key=lambda j: np.sqrt((j[0]-th)**2+(j[1]-th)**2))
                        index = local_mask[candidates[0]]-2

                        # add pixel to mask/bbox
                        # x,y to bbox with index
                        if x < r['rois'][index][0]:
                            r['rois'][index][0] = x

                        elif x > r['rois'][index][2]:
                            r['rois'][index][2] = x

                        if y < r['rois'][index][1]:
                            r['rois'][index][1] = y

                        elif y > r['rois'][index][3]:
                            r['rois'][index][3] = y

                        # update the mask
                        mask[x, y] = index + 2

        # resolving overlapping problem
        bbox_dict = {}  # to check any overlapping bbox
        class_id_check = []

        for i in range(len(r['rois'])):
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]

            region_bbox = [min_y, min_x, max_y, max_x]

            for key in bbox_dict:
                for bbox in bbox_dict[key]:

                    # checking for ymax case with vertical overlapping
                    # along with y, check both for xmax and xmin
                    if (region_bbox[3] <= bbox[3] and region_bbox[3] >= bbox[1] and
                        ((region_bbox[0] >= bbox[0] and region_bbox[0] <= bbox[2]) or (region_bbox[2] >= bbox[0]
                                                                                       and region_bbox[2] <= bbox[2]) or (region_bbox[0] <= bbox[0] and region_bbox[2] >= bbox[2]))
                            and r['class_ids'][i] != 5):

                        r['rois'][i][2] = bbox[1] - 1

                    # checking for ymin now
                    # along with y, check both for xmax and xmin
                    if (region_bbox[1] <= bbox[3] and region_bbox[1] >= bbox[1] and
                        ((region_bbox[0] >= bbox[0] and region_bbox[0] <= bbox[2]) or (region_bbox[2] >= bbox[0]
                                                                                       and region_bbox[2] <= bbox[2]) or (region_bbox[0] <= bbox[0] and region_bbox[2] >= bbox[2]))
                            and r['class_ids'][i] != 5):

                        r['rois'][i][0] = bbox[3] + 1

            if r['class_ids'][i] not in class_id_check:
                bbox_dict[r['class_ids'][i]] = []
                class_id_check.append(r['class_ids'][i])

            bbox_dict[r['class_ids'][i]].append(region_bbox)

        # resolving overlapping problem code

        # define reading order on basis of coordinates
        reading_order = []

        for i in range(len(r['rois'])):
            width, height, _ = img_array.shape
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]

            if (min_y - 5) > width and r['class_ids'][i] == 2:
                min_y -= 5
            if (max_y + 10) < width and r['class_ids'][i] == 2:
                min_y += 10
            reading_order.append((min_y, min_x, max_y, max_x))

        reading_order = sorted(reading_order, key=lambda reading_order: (reading_order[1], reading_order[0]))
        for i in range(len(reading_order)):
            min_y, min_x, max_y, max_x = reading_order[i]
            min_y = 0
            i_poly = Polygon([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            for j in range(i+1, len(reading_order)):
                min_y, min_x, max_y, max_x = reading_order[j]
                j_poly = Polygon([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
                inter = i_poly.intersection(j_poly)
                if inter:
                    reading_order.insert(j+1, reading_order[i])
                    del reading_order[i]

        # Creating Reading Order object in PageXML
        order_group = OrderedGroupType(caption="Regions reading order", id=page_id)

        for i in range(len(r['rois'])):
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]
            if (min_y - 5) > width and r['class_ids'][i] == 2:
                min_y -= 5
            if (max_y + 10) < width and r['class_ids'][i] == 2:
                min_y += 10

            region_polygon = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            if border:
                cut_region_polygon = border.intersection(Polygon(region_polygon))
                if cut_region_polygon.is_empty:
                    continue
            else:
                cut_region_polygon = Polygon(region_polygon)

            order_index = reading_order.index((min_y, min_x, max_y, max_x))
            region_id = '%s_region%04d' % (page_id, i)
            regionRefIndex = RegionRefIndexedType(index=order_index, regionRef=region_id)
            order_group.add_RegionRefIndexed(regionRefIndex)

        reading_order_object = ReadingOrderType()
        reading_order_object.set_OrderedGroup(order_group)
        page.set_ReadingOrder(reading_order_object)

        for i in range(len(r['rois'])):
            width, height, _ = img_array.shape
            min_x = r['rois'][i][0]
            min_y = r['rois'][i][1]
            max_x = r['rois'][i][2]
            max_y = r['rois'][i][3]

            if (min_y - 5) > width and r['class_ids'][i] == 2:
                min_y -= 5
            if (max_y + 10) < width and r['class_ids'][i] == 2:
                min_y += 10

            # one change here to resolve flipped coordinates
            region_polygon = [[min_y, min_x], [max_y, min_x], [max_y, max_x], [min_y, max_x]]

            cut_region_polygon = border.intersection(Polygon(region_polygon))

            if cut_region_polygon.is_empty:
                continue
            cut_region_polygon = [j for j in zip(list(cut_region_polygon.exterior.coords.xy[0]), list(cut_region_polygon.exterior.coords.xy[1]))][:-1]

            # checking whether coordinates are flipped

            region_polygon = coordinates_for_segment(cut_region_polygon, page_image, page_xywh)
            region_points = points_from_polygon(region_polygon)

            read_order = reading_order.index((min_y, min_x, max_y, max_x))

            # this can be tested, provided whether we need previous comments or not?
            # resolving overlapping problem

            region_img = img_array[min_x:max_x, min_y:max_y]  # extract from points and img_array

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

            # incase of imageRegion
            if r['class_ids'][i] == 15:
                image_region = ImageRegionType(custom='readingOrder {index:'+str(read_order)+';}', id=region_id, Coords=coords, type_=class_names[r['class_ids'][i]])
                # image_region.add_AlternativeImage(ai)
                page.add_ImageRegion(image_region)
                continue
            if r['class_ids'][i] == 16:
                table_region = TableRegionType(custom='readingOrder {index:'+str(read_order)+';}', id=region_id, Coords=coords, type_=class_names[r['class_ids'][i]])
                # table_region.add_AlternativeImage(ai)
                page.add_TableRegion(table_region)
                continue
            if r['class_ids'][i] == 17:
                graphic_region = GraphicRegionType(custom='readingOrder {index:'+str(read_order)+';}', id=region_id, Coords=coords, type_=class_names[r['class_ids'][i]])
                # graphic_region.add_AlternativeImage(ai)
                page.add_GraphicRegion(graphic_region)
                continue

            textregion = TextRegionType(custom='readingOrder {index:'+str(read_order)+';}', id=region_id, Coords=coords, type_=class_names[r['class_ids'][i]])
            # textregion.add_AlternativeImage(ai)

            #border = page.get_Border()
            # if border:
            #    border.add_TextRegion(textregion)
            # else:
            page.add_TextRegion(textregion)


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrBlockSegmenter, *args, **kwargs)
