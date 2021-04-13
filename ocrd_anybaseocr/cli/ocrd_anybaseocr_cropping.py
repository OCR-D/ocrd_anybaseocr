# pylint: disable=invalid-name
# ======================================================================
# ====================================
# README file for Page Cropping component
# ====================================
# Filename : ocrd-anyBaseOCR-pagecropping.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and crops/selects the page content
# area only (that's mean remove textual noise as well as any other noise around page content area)
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright
# and license notices. Contributors provide an express grant of patent rights.
# Licensed works, modifications, and larger works may be distributed under
# different terms and without source code.

# *********** LICENSE ********************
# ======================================================================


import os
import numpy as np
from pylsd.lsd import lsd
import cv2
from PIL import Image

import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor


from ..constants import OCRD_TOOL
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_utils import (
    getLogger,
    crop_image,
    make_file_id,
    assert_file_grp_cardinality,
    concat_padded, 
    MIMETYPE_PAGE, 
    coordinates_for_segment,
    coordinates_of_segment,
    bbox_from_points,
    bbox_from_polygon,
    points_from_polygon,
    polygon_from_bbox
)
from ocrd_models.ocrd_page import (
    CoordsType,
    AlternativeImageType,
    to_xml,
)
from ocrd_models.ocrd_page_generateds import BorderType

TOOL = 'ocrd-anybaseocr-crop'

# enable to plot intermediate results interactively:
DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch

# originally from ocrolib (here also with alpha support):
def pil2array(im,alpha=0):
    if im.mode=="L":
        a = np.fromstring(im.tobytes(),'B')
        a.shape = im.height, im.width
        return a
    if im.mode=="LA":
        a = np.fromstring(im.tobytes(),'B')
        a.shape = im.height, im.width, 2
        if not alpha: a = a[:,:,0]
        return a
    if im.mode=="RGB":
        a = np.fromstring(im.tobytes(),'B')
        a.shape = im.height, im.width, 3
        return a
    if im.mode=="RGBA":
        a = np.fromstring(im.tobytes(),'B')
        a.shape = im.height, im.width, 4
        if not alpha: a = a[:,:,:3]
        return a
    # fallback to Pillow grayscale conversion:
    return pil2array(im.convert("L"))

class OcrdAnybaseocrCropper(Processor):

    def __init__(self, *args, **kwargs):

        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrCropper, self).__init__(*args, **kwargs)

    def remove_ruler(self, arg):
        gray = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = arg.shape
        imgArea = height*width

        # Get bounding box x,y,w,h of each contours
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: (x[2]*x[3]), reverse=True)
        # consider those rectangle whose area>10000 and less than one-fourth of images
        rects = [r for r in rects if (
            imgArea*self.parameter['rulerAreaMax']) > (r[2]*r[3]) > (imgArea*self.parameter['rulerAreaMin'])]

        # detect child rectangles. Usually those are not ruler. Ruler position are basically any one side.
        removeRect = []
        for i, rect1 in enumerate(rects):
            (x1, y1, w1, h1) = rect1
            for rect2 in rects[i+1:len(rects)]:
                (x2, y2, w2, h2) = rect2
                if (x1 < x2) and (y1 < y2) and (x1+w1 > x2+w2) and (y1+h1 > y2+h2):
                    removeRect.append(rect2)

        # removed child rectangles.
        rects = [x for x in rects if x not in removeRect]

        predictRuler = []

        y1max = self.parameter['marginTop'] * height
        y2min = self.parameter['marginBottom'] * height
        x1max = self.parameter['marginLeft'] * width
        x2min = self.parameter['marginRight'] * width
        wmax = self.parameter['rulerWidthMax'] * width
        for rect in rects:
            (x, y, w, h) = rect
            if (w < wmax and
                ((y+h < y1max or y > y2min) or
                 (x+w < x1max or x > x2min))):
                if (self.parameter['rulerRatioMin'] < round(float(w)/float(h), 2) < self.parameter['rulerRatioMax']) or \
                   (self.parameter['rulerRatioMin'] < round(float(h)/float(w), 2) < self.parameter['rulerRatioMax']):
                    blackPixel = np.count_nonzero(arg[y:y+h, x:x+w] == 0)
                    predictRuler.append((x, y, w, h, blackPixel))

        # Finally check number of black pixel to avoid false ruler
        if predictRuler:
            # sort by fg size
            predictRuler = sorted(predictRuler, key=lambda x: x[4], reverse=True)
            # pick largest candidate
            x, y, w, h, _ = predictRuler[0]
            # clip to white
            cv2.rectangle(arg, (x-15, y-15), (x+w+20, y+h+20),
                          (255, 255, 255), cv2.FILLED)
            if DEBUG:
                plt.imshow(arg)
                plt.legend(handles=[Patch(label='ruler clipped to white')])
                plt.show()

        return arg

    def get_intersect(self, a1, a2, b1, b2):
        s = np.vstack([a1, a2, b1, b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)
        if z == 0:
            # return (float('inf'), float('inf'))
            return (0, 0)
        return (x/z, y/z)

    def get_area(self, box):
        x1, y1, x2, y2 = box
        return abs(y2 - y1) * abs(x2 - x1)

    def detect_lines(self, arg):
        gray = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        lines = lsd(gray)
        if DEBUG:
            plt.imshow(arg)
            for x1, y1, x2, y2, _ in lines:
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=2, linestyle='dashed'))
            plt.legend(handles=[Patch(label='line segments')])
            plt.show()
        imgHeight, imgWidth, _ = arg.shape
        y1max = self.parameter['marginTop'] * imgHeight
        y2min = self.parameter['marginBottom'] * imgHeight
        x1max = self.parameter['marginLeft'] * imgWidth
        x2min = self.parameter['marginRight'] * imgWidth
        hlines = []
        vlines = []
        for x1, y1, x2, y2, _ in lines:
            # consider line segments near margins and not too short
            if abs(x1 - x2) > 45 and (y1 < y1max or y2 > y2min): #y1 > y2min
                # make full horizontal line
                hlines.append([0, int(y1), imgWidth, int(y2)])
            if abs(y1 - y2) > 45 and (x1 < x1max or x2 > x2min): #x1 > x2min
                # make full vertical line
                vlines.append([int(x1), 0, int(x2), imgHeight])
        hlines.sort(key=lambda p: p[1]) # from top to bottom
        vlines.sort(key=lambda p: p[0]) # from left to right
        return imgHeight, imgWidth, hlines, vlines

    def select_borderLine(self, arg, lineDetectH, lineDetectV):
        imgHeight, imgWidth, Hlines, Vlines = self.detect_lines(arg)
        y1max = self.parameter['marginTop'] * imgHeight
        y2min = self.parameter['marginBottom'] * imgHeight
        x1max = self.parameter['marginLeft'] * imgWidth
        x2min = self.parameter['marginRight'] * imgWidth
        # connect line segments on all margins
        self.BorderLine(y1max, Hlines, "top", lineDetectH)
        self.BorderLine(x1max, Vlines, "left", lineDetectV)
        self.BorderLine(y2min, Hlines, "bottom", lineDetectH)
        self.BorderLine(x2min, Vlines, "right", lineDetectV)
        if DEBUG:
            plt.imshow(arg)
            for x1, y1, x2, y2 in lineDetectH+lineDetectV:
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=2, linestyle='dotted'))
            plt.legend(handles=[Patch(label='border lines')])
            plt.show()

        intersectPoint = []
        for hx1, hy1, hx2, hy2 in lineDetectH:
            for vx1, vy1, vx2, vy2 in lineDetectV:
                x, y = self.get_intersect((hx1, hy1),
                                          (hx2, hy2),
                                          (vx1, vy1),
                                          (vx2, vy2))
                intersectPoint.append([x, y])
        Xstart = 0
        Xend = imgWidth
        Ystart = 0
        Yend = imgHeight
        for xi, yi in intersectPoint:
            Xs = int(xi)+10 if xi < x1max else 10
            if Xs > Xstart:
                Xstart = Xs
            Xe = int(xi)-10 if xi > x2min else int(imgWidth)-10
            if Xe < Xend:
                Xend = Xe
            Ys = int(yi)+10 if yi < y1max else 10
            # print("Ys,Ystart:",Ys,Ystart)
            if Ys > Ystart:
                Ystart = Ys
            Ye = int(yi)-15 if yi > y2min else int(imgHeight)-15
            if Ye < Yend:
                Yend = Ye

        if Xend < 0:
            Xend = 10
        if Yend < 0:
            Yend = 15

        return [Xstart, Ystart, Xend, Yend]

    def BorderLine(self, boundary, lines, side, detected):
        ncontiguous = 1
        lastline = []
        if side in ('top', 'bottom'):
            index = 1 # get y
        else:
            index = 0 # get x
        if side in ('top', 'left'):
            for line, nextline in zip(lines[:-1], lines[1:]):
                if(abs(line[index]-nextline[index])) <= 15 and line[index] < boundary:
                    lastline = line
                    ncontiguous += 1
                elif ncontiguous >= 3:
                    break
                else:
                    ncontiguous = 1
        else:
            for line, nextline in list(zip(lines[:-1], lines[1:]))[::-1]:
                if(abs(line[index]-nextline[index])) <= 15 and line[index] > boundary:
                    lastline = line
                    ncontiguous += 1
                elif ncontiguous >= 3:
                    break
                else:
                    ncontiguous = 1
        if ncontiguous >= 3 and lastline:
            if side == "top":
                y = max(lastline[1], lastline[3])
                detected.append((lastline[0], y, lastline[2], y))
            elif side == "left":
                x = max(lastline[0], lastline[2])
                detected.append((x, lastline[1], x, lastline[3]))
            elif side == "bottom":
                y = min(lastline[1], lastline[3])
                detected.append((lastline[0], y, lastline[2], y))
            else:
                x = min(lastline[0], lastline[2])
                detected.append((x, lastline[1], x, lastline[3]))

    def filter_noisebox(self, textboxes, height, width):
        tmp = []
        st = True
        minArea = height * width * 0.001

        while st:
            textboxes = [list(x) for x in textboxes
                         if x not in tmp]
            if len(textboxes) > 1:
                tmp = []
                textboxes = sorted(textboxes, key=lambda x: x[3])
                # print textarea
                x11, y11, x12, y12 = textboxes[0]
                x21, y21, x22, y22 = textboxes[1]

                if abs(y12-y21) > 100 and self.get_area(textboxes[0]) < minArea:
                    tmp.append(textboxes[0])

                x11, y11, x12, y12 = textboxes[-2]
                x21, y21, x22, y22 = textboxes[-1]

                if abs(y12-y21) > 100 and self.get_area(textboxes[-1]) < minArea:
                    tmp.append(textboxes[-1])

                if len(tmp) == 0:
                    st = False
            else:
                break

        return textboxes

    def detect_textboxes(self, arg):
        textboxes = []
        small = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        height, width, _ = arg.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
        if DEBUG:
            plt.imshow(grad)
            plt.legend(handles=[Patch(label='morphological gradient')])
            plt.show()

        _, bw = cv2.threshold(
            grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if DEBUG:
            plt.imshow(bw)
            plt.legend(handles=[Patch(label='binarized gradient')])
            plt.show()

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (10, 1))  # for historical docs
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        if DEBUG:
            plt.imshow(connected)
            plt.legend(handles=[Patch(label='horizontal closing')])
            plt.show()

        contours, hierarchy = cv2.findContours(
            connected.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if DEBUG: mask = np.zeros(bw.shape[:2], dtype=np.uint16)

        def apply_contour(idx):
            while 0 <= idx < len(contours):
                next_idx, prev_idx, child_idx, parent_idx = hierarchy[0, idx].tolist()
                if len(contours[idx]) >= 3:
                    x, y, w, h = cv2.boundingRect(contours[idx])
                    if DEBUG: cv2.drawContours(mask, contours, idx, idx+1, -1)
                    r = cv2.contourArea(contours[idx]) / (w * h)
                    if r > 0.45 and (width*0.9) > w > 15 and (height*0.5) > h > 15:
                        textboxes.append([x, y, x+w-1, y+h-1])
                        if DEBUG: cv2.rectangle(arg, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
                    if child_idx >= 0:
                        apply_contour(child_idx)
                idx = next_idx
        if contours:
            apply_contour(0)
        if DEBUG:
            mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            plt.imshow(mask)
            plt.legend(handles=[Patch(label='contours')])
            plt.show()

        if len(textboxes) > 1:
            textboxes = self.filter_noisebox(textboxes, height, width)
        if DEBUG:
            plt.imshow(arg)
            plt.legend(handles=[Patch(label='text boxes')])
            plt.show()

        return textboxes, height, width

    def merge_columns(self, textboxes, colSeparator):
        textboxes.sort(key=lambda x: x[0]) # f.l.t.r.
        columns = [textboxes[0]]
        for box in textboxes[1:]:
            x11, y11, x12, y12 = columns[-1]
            x21, y21, x22, y22 = box
            if x21-x12 <= colSeparator:
                columns[-1] = (min(x11, x21), min(y11, y21),
                               max(x12, x22), max(y12, y22))
            else:
                columns.append(box)
        return columns

    def merge_boxes(self, textboxes, img, colSeparator):
        height, width, _ = img.shape

        textboxes = np.unique(textboxes, axis=0)
        i = 0
        tmp = []
        boxes = []
        while i < len(textboxes):
            textboxes = [list(x) for x in textboxes
                         if x not in tmp]
            tmp = []
            if len(textboxes) == 0:
                break
            maxBox = textboxes[0]
            for chkBox in textboxes[1:]:
                x11, y11, x12, y12 = maxBox
                x21, y21, x22, y22 = chkBox
                if ((x11 <= x21 <= x12) or (x21 <= x11 <= x22)):
                    tmp.append(maxBox)
                    tmp.append(chkBox)
                    maxBox = [min(x11, x21), min(y11, y21),
                              max(x12, x22), max(y12, y22)]
            if len(tmp) == 0:
                tmp.append(maxBox)
            boxes.append(maxBox)
            i = i+1
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in boxes:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            plt.legend(handles=[Patch(label='columns')])
            plt.show()

        columns = np.unique(boxes, axis=0).tolist()
        if len(columns) > 0:
            minArea = height * width * self.parameter['columnAreaMin']
            columns = list(filter(lambda box: self.get_area(box) > minArea, columns))
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in columns:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            plt.legend(handles=[Patch(label='minArea columns')])
            plt.show()
        if len(columns) > 1:
            columns = self.merge_columns(columns, colSeparator)

        if len(columns) > 0:
            columns = sorted(columns, key=self.get_area)
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in columns:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            plt.legend(handles=[Patch(label='merged columns')])
            plt.show()

        return columns

    def process(self):
        """Performs border detection on the workspace. """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('OcrdAnybaseocrCropper')

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            # Check for existing Border --> already cropped
            border = page.get_Border()
            if border:
                left, top, right, bottom = bbox_from_points(
                    border.get_Coords().points)
                LOG.warning('Overwriting existing Border: %i:%i,%i:%i',
                            left, top, right, bottom)

            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, # should also be deskewed
                feature_filter='cropped,binarized')

            self._process_page(page, page_image, page_coords, input_file)
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

    def _process_page(self, page, page_image, page_xywh, input_file):
        img_array = pil2array(page_image)

        # ensure RGB image
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)

        img_array_rr = self.remove_ruler(img_array)
        textboxes, height, width = self.detect_textboxes(img_array_rr)

        lineDetectH = []
        lineDetectV = []
        colSeparator = int(width * self.parameter['columnSepWidthMax'])
        if len(textboxes) > 1:
            textboxes = self.merge_boxes(textboxes, img_array, colSeparator)

            if len(textboxes) == 0:
                min_x, min_y, max_x, max_y = self.select_borderLine(
                    img_array_rr, lineDetectH, lineDetectV)
            else:
                min_x, min_y, max_x, max_y = textboxes[0]
        elif len(textboxes) == 1 and (height*width*0.5 < self.get_area(textboxes[0])):
            min_x, min_y, max_x, max_y = textboxes[0]
        else:
            min_x, min_y, max_x, max_y = self.select_borderLine(
                img_array_rr, lineDetectH, lineDetectV)

        def clip(point):
            x, y = point
            x = max(0, min(page_image.width, x))
            y = max(0, min(page_image.height, y))
            return x, y
        pad = self.parameter['padding']
        border_polygon = polygon_from_bbox(min_x - pad, min_y - pad, max_x + pad, max_y + pad)
        border_polygon = coordinates_for_segment(border_polygon, page_image, page_xywh)
        border_polygon = list(map(clip, border_polygon))
        border_points = points_from_polygon(border_polygon)
        border = BorderType(Coords=CoordsType(border_points))
        page.set_Border(border)
        # get clipped relative coordinates for current image
        border_polygon = coordinates_of_segment(border, page_image, page_xywh)
        border_bbox = bbox_from_polygon(border_polygon)

        page_image = crop_image(page_image, box=border_bbox)
        page_xywh['features'] += ',cropped'

        file_id = make_file_id(input_file, self.output_file_grp)
        file_path = self.workspace.save_image_file(page_image,
                                                   file_id + '-IMG',
                                                   page_id=input_file.pageId,
                                                   file_grp=self.output_file_grp)
        page.add_AlternativeImage(AlternativeImageType(
            filename=file_path, comments=page_xywh['features']))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrCropper, *args, **kwargs)
