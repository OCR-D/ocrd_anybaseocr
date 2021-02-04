# pylint: disable=missing-module-docstring, missing-class-docstring, invalid-name
# pylint: disable=line-too-long, import-error, no-name-in-module, too-many-statements
# pylint: disable=wrong-import-position, wrong-import-order, too-many-locals, too-few-public-methods
import sys
import os
from pathlib import Path
from pkg_resources import resource_filename

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
    make_file_id,
    assert_file_grp_cardinality,
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
    RegionRefIndexedType, OrderedGroupType, ReadingOrderType
)
from ..mrcnn import model
from ..mrcnn.config import Config
from ..constants import OCRD_TOOL
from ..tensorflow_importer import tf

TOOL = 'ocrd-anybaseocr-block-segmentation'
CLASS_NAMES = ['BG',
               'page-number',
               'paragraph',
               'catch-word',
               'heading',
               'drop-capital',
               'signature-mark',
               'header',
               'marginalia',
               'footnote',
               'footnote-continued',
               'caption',
               'endnote',
               'footer',
               'keynote',
               # not included in the provided models yet:
               #'image',
               #'table',
               #'graphics'
]

class InferenceConfig(Config):

    def __init__(self, confidence):
        Config.__init__(self, confidence)

    NAME = "block"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

#     NUM_CLASSES = 1 + 14
#     DETECTION_MIN_CONFIDENCE = 0.9 # needs to be changed back to parameter

class OcrdAnybaseocrBlockSegmenter(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrBlockSegmenter, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp') and hasattr(self, 'parameter'):
            # processing context
            self.setup()

    def setup(self):
        LOG = getLogger('processor.AnybaseocrBlockSegmenter')
        #self.reading_order = []
        self.order = 0
        model_path = resource_filename(__name__, '../mrcnn')
        model_weights = Path(self.resolve_resource(self.parameter['block_segmentation_weights']))

        confidence = self.parameter['min_confidence']
        config = InferenceConfig(confidence)
        self.mrcnn_model = model.MaskRCNN(mode="inference", model_dir=str(model_path), config=config)
        self.mrcnn_model.load_weights(str(model_weights), by_name=True)
    
    def process(self):
        """Segment pages into regions using a Mask R-CNN model."""
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('processor.AnybaseocrBlockSegmenter')
        if not tf.test.is_gpu_available():
            LOG.warning("Tensorflow cannot detect CUDA installation. Running without GPU will be slow.")

        for input_file in self.input_files:
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            page_id = input_file.pageId or input_file.ID

            # todo rs: why not cropped?
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_filter='binarized,deskewed,cropped,clipped,non_text')
            # try to load pixel masks
            try:
                # todo rs: this combination only works for tiseg with use_deeplr=true
                mask_image, _, _ = self.workspace.image_from_page(page, page_id, feature_selector='clipped', feature_filter='binarized,deskewed,cropped,non_text')
            except:
                mask_image = None
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None

            self._process_segment(page_image, page, page_xywh, page_id, input_file, mask_image, dpi)

            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts).encode('utf-8')
            )

    def _process_segment(self, page_image, page, page_xywh, page_id, input_file, mask, dpi):
        LOG = getLogger('processor.AnybaseocrBlockSegmenter')
        # check for existing text regions and whether to overwrite them
        if page.get_TextRegion() or page.get_TableRegion():
            if self.parameter['overwrite']:
                LOG.info('removing existing text/table regions in page "%s"', page_id)
                page.set_TextRegion([])
            else:
                LOG.warning('keeping existing text/table regions in page "%s"', page_id)
        # check if border exists
        border_polygon = None
        if page.get_Border():
            border_coords = page.get_Border().get_Coords()
            border_points = polygon_from_points(border_coords.get_points())
            border_polygon = Polygon(border_points)

        LOG.info('detecting regions on page "%s"', page_id)
        img_array = ocrolib.pil2array(page_image)
        if len(img_array.shape) <= 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        # convert to incidence matrix
        class_ids = np.array([[1 if category in self.parameter['active_classes'] else 0
                               for category in CLASS_NAMES]], dtype=np.int32)
        results = self.mrcnn_model.detect([img_array], verbose=0, active_class_ids=class_ids)
        r = results[0]
        LOG.info('found %d candidates on page "%s"', len(r['rois']), page_id)

        th = self.parameter['th']
        # check for existing semgentation mask
        # this code executes only when the workflow had tiseg run before with use_deeplr=true
        if mask:
            mask = ocrolib.pil2array(mask)
            mask = mask//255
            mask = 1-mask
            # multiply all the bounding box part with 2
            for i in range(len(r['rois'])):

                min_y, min_x, max_y, max_x = r['rois'][i]
                mask[min_y:max_y, min_x:max_x] *= i+2

            # check for left over pixels and add them to the bounding boxes
            pixel_added = True

            while pixel_added:

                pixel_added = False
                left_over = np.where(mask == 1)
                for y, x in zip(left_over[0], left_over[1]):
                    local_mask = mask[y-th:y+th, x-th:x+th]
                    candidates = np.where(local_mask > 1)
                    candidates = [k for k in zip(candidates[0], candidates[1])]
                    if len(candidates) > 0:
                        pixel_added = True
                        # find closest pixel with x>1
                        candidates.sort(key=lambda j: np.sqrt((j[0]-th)**2+(j[1]-th)**2))
                        index = local_mask[candidates[0]]-2

                        # add pixel to mask/bbox
                        # y,x to bbox with index
                        if y < r['rois'][index][0]:
                            r['rois'][index][0] = y

                        elif y > r['rois'][index][2]:
                            r['rois'][index][2] = y

                        if x < r['rois'][index][1]:
                            r['rois'][index][1] = x

                        elif x > r['rois'][index][3]:
                            r['rois'][index][3] = x

                        # update the mask
                        mask[y, x] = index + 2

        for i in range(len(r['rois'])):
            class_id = r['class_ids'][i]
            if class_id >= len(CLASS_NAMES):
                raise Exception('Unexpected class id %d - model does not match' % class_id)

        # find hull contours on masks
        if self.parameter['use_masks']:
            r.setdefault('polygons', list())
            # estimate glyph scale (roughly)
            scale = int(dpi / 6)
            scale = scale + (scale+1)%2 # odd
            for i in range(len(r['rois'])):
                mask = r['masks'][:,:,i]
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((scale,scale), np.uint8)) > 0
                # close mask until we have a single outer contour
                contours = None
                for _ in range(10):
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                            np.ones((scale,scale), np.uint8)) > 0
                    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) == 1:
                        break
                r['polygons'].append(Polygon(contours[0][:,0,:])) # already in x,y order

        # to reduce overlaps, apply IoU-based non-maximum suppression
        # (and other post-processing against overlaps) across classes,
        # but not on the raw pixels, but the smoothed hull polygons
        LOG.info('post-processing detections on page "%s"', page_id)
        worse = []
        if self.parameter['post_process']:
            active = True
            def _merge_rois(i, j):
                """merges i into j"""
                nonlocal r, active
                r['rois'][j][0] = min(r['rois'][i][0], r['rois'][j][0])
                r['rois'][j][1] = min(r['rois'][i][1], r['rois'][j][1])
                r['rois'][j][2] = max(r['rois'][i][2], r['rois'][j][2])
                r['rois'][j][3] = max(r['rois'][i][3], r['rois'][j][3])
                r['polygons'][j] = r['polygons'][i].union(r['polygons'][j])
                #r['scores'][j] = max(r['scores'][i], r['scores'][i])
                active = True
            # find overlapping pairs
            while active:
                active = False
                for i in range(len(r["class_ids"])):
                    if i in worse:
                        continue
                    for j in range(i + 1, len(r['class_ids'])):
                        if j in worse:
                            continue
                        iclass = r['class_ids'][i]
                        jclass = r['class_ids'][j]
                        iname = CLASS_NAMES[iclass]
                        jname = CLASS_NAMES[jclass]
                        if (iname == 'drop-capital') != (jname == 'drop-capital'):
                            # ignore drop-capital overlapping with others
                            continue
                        # rs todo: lower priority for footnote?
                        if (r['rois'][i][1] > r['rois'][j][3] or
                            r['rois'][i][3] < r['rois'][j][1] or
                            r['rois'][i][0] > r['rois'][j][2] or
                            r['rois'][i][2] < r['rois'][j][0]):
                            # no overlap (cut)
                            continue
                        iscore = r['scores'][i]
                        jscore = r['scores'][j]
                        if not self.parameter['use_masks']:
                            LOG.debug("roi %d[%s] overlaps roi %d[%s] and %s (replacing)",
                                      i, iname, j, jname,
                                      "looses" if iscore < jscore else "wins")
                            if iscore < jscore:
                                worse.append(i)
                                break
                            else:
                                worse.append(j)
                                continue
                        # compare masks
                        ipoly = r['polygons'][i]
                        jpoly = r['polygons'][j]
                        isize = ipoly.area
                        jsize = jpoly.area
                        inter = ipoly.intersection(jpoly).area
                        union = ipoly.union(jpoly).area
                        # LOG.debug("%d/%d %dpx/%dpx shared %dpx overall %dpx",
                        #           i, j, isize, jsize, inter, union)
                        if inter / isize > self.parameter['min_share_drop']:
                            LOG.debug("roi %d[%s] contains roi %d[%s] (replacing)",
                                      j, jname, i, iname)
                            worse.append(i)
                            break
                        elif inter / jsize > self.parameter['min_share_drop']:
                            LOG.debug("roi %d[%s] contains roi %d[%s] (replacing)",
                                      i, iname, j, jname)
                            worse.append(j)
                        elif inter / union > self.parameter['min_iou_drop']:
                            LOG.debug("roi %d[%s] heavily overlaps roi %d[%s] and %s (replacing)",
                                      i, iname, j, jname,
                                      "looses" if iscore < jscore else "wins")
                            if iscore < jscore:
                                worse.append(i)
                                break
                            else:
                                worse.append(j)
                        elif inter / isize > self.parameter['min_share_merge']:
                            LOG.debug("roi %d[%s] covers roi %d[%s] (merging)",
                                      j, jname, i, iname)
                            worse.append(i)
                            _merge_rois(i, j)
                            break
                        elif inter / jsize > self.parameter['min_share_merge']:
                            LOG.debug("roi %d[%s] covers roi %d[%s] (merging)",
                                      i, iname, j, jname)
                            worse.append(j)
                            _merge_rois(j, i)
                        elif inter / union > self.parameter['min_iou_merge']:
                            LOG.debug("roi %d[%s] slightly overlaps roi %d[%s] and %s (merging)",
                                      i, iname, j, jname,
                                      "looses" if iscore < jscore else "wins")
                            if iscore < jscore:
                                worse.append(i)
                                _merge_rois(i, j)
                                break
                            else:
                                worse.append(j)
                                _merge_rois(j, i)

        # define reading order on basis of coordinates
        partial_order = np.zeros((len(r['rois']), len(r['rois'])), np.uint8)
        for i, (min_y_i, min_x_i, max_y_i, max_x_i) in enumerate(r['rois']):
            for j, (min_y_j, min_x_j, max_y_j, max_x_j) in enumerate(r['rois']):
                if min_x_i < max_x_j and max_x_i > min_x_j:
                    # xoverlaps
                    if min_y_i < min_y_j:
                        partial_order[i, j] = 1
                else:
                    min_y = min(min_y_i, min_y_j)
                    max_y = max(max_y_i, max_y_j)
                    min_x = min(min_x_i, min_x_j)
                    max_x = max(max_x_i, max_x_j)
                    if next((False for (min_y_k, min_x_k, max_y_k, max_x_k) in r['rois']
                             if (min_y_k < max_y and max_y_k > min_y and
                                 min_x_k < max_x and max_x_k > min_x)),
                            True):
                        # no k in between
                        if ((min_y_j + max_y_j)/2 < min_y_i and
                            (min_y_i + max_y_i)/2 > max_y_j):
                            # vertically unrelated
                            partial_order[j, i] = 1
                        elif max_x_i < min_x_j:
                            partial_order[i, j] = 1
        def _topsort(po):
            visited = np.zeros(po.shape[0], np.bool)
            result = list()
            def _visit(k):
                if visited[k]:
                    return
                visited[k] = True
                for l in np.nonzero(po[:, k])[0]:
                    _visit(l)
                result.append(k)
            for k in range(po.shape[0]):
                _visit(k)
            return result
        reading_order = _topsort(partial_order)

        # Creating Reading Order object in PageXML
        order_group = OrderedGroupType(caption="Regions reading order", id=page_id)
        reading_order_object = ReadingOrderType()
        reading_order_object.set_OrderedGroup(order_group)
        page.set_ReadingOrder(reading_order_object)

        for i in range(len(r['rois'])):
            width, height, _ = img_array.shape
            min_y, min_x, max_y, max_x = r['rois'][i]
            score = r['scores'][i]
            class_id = r['class_ids'][i]
            class_name = CLASS_NAMES[class_id]
            if i in worse:
                LOG.debug("Ignoring instance %d[%s] overlapping better/larger neighbour",
                          i, class_name)
                continue

            if self.parameter['use_masks']:
                region_polygon = r['polygons'][i].exterior.coords[:-1]
            else:
                region_polygon = polygon_from_bbox(
                    max(min_x - 5, 0) if class_name == 'paragraph' else min_x,
                    min_y,
                    min(max_x + 10, width) if class_name == 'paragraph' else max_x,
                    max_y)

            # convert to absolute coordinates
            region_polygon = coordinates_for_segment(region_polygon, page_image, page_xywh)
            # intersect with parent and plausibilize
            cut_region_polygon = Polygon(region_polygon)
            if border_polygon:
                cut_region_polygon = border_polygon.intersection(cut_region_polygon)
            if cut_region_polygon.is_empty:
                LOG.warning('region %d does not intersect page frame', i)
                continue
            if not cut_region_polygon.is_valid:
                LOG.warning('region %d has invalid polygon', i)
                continue
            region_polygon = cut_region_polygon.exterior.coords[:-1]
            region_coords = CoordsType(points_from_polygon(region_polygon),
                                       conf=score)
            read_order = reading_order.index(i)
            region_args = {'custom': 'readingOrder {index:'+str(read_order)+';}',
                           'id': 'region%04d' % i,
                           'Coords': region_coords}
            if class_name == 'image':
                image_region = ImageRegionType(**region_args)
                page.add_ImageRegion(image_region)
            elif class_name == 'table':
                table_region = TableRegionType(**region_args)
                page.add_TableRegion(table_region)
            elif class_name == 'graphics':
                graphic_region = GraphicRegionType(**region_args)
                page.add_GraphicRegion(graphic_region)
            else:
                region_args['type_'] = class_name
                textregion = TextRegionType(**region_args)
                page.add_TextRegion(textregion)
            order_index = reading_order.index(i)
            regionRefIndex = RegionRefIndexedType(index=order_index, regionRef=region_args['id'])
            order_group.add_RegionRefIndexed(regionRefIndex)
            LOG.info('added %s region on page "%s"', class_name, page_id)


@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrBlockSegmenter, *args, **kwargs)
