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
import ocrolib
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
    concat_padded, 
    MIMETYPE_PAGE, 
    coordinates_for_segment,
    points_from_polygon
)
from ocrd_models.ocrd_page import (
    CoordsType,
    AlternativeImageType,
    to_xml,
    MetadataItemType,
    LabelsType, LabelType,
)
from ocrd_models.ocrd_page_generateds import BorderType

TOOL = 'ocrd-anybaseocr-crop'

LOG = getLogger('OcrdAnybaseocrCropper')
FALLBACK_IMAGE_GRP = 'OCR-D-IMG-CROP'


class OcrdAnybaseocrCropper(Processor):

    def __init__(self, *args, **kwargs):

        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrCropper, self).__init__(*args, **kwargs)

    def write_crop_coordinate(self, base, coordinate):
        x1, y1, x2, y2 = coordinate
        with open(base + '-frame-pf.dat', 'w') as fp:
            fp.write(str(x1)+"\t"+str(y1)+"\t"+str(x2-x1)+"\t"+str(y2-y1))

    def rotate_image(self, orientation, image):
        return image.rotate(orientation)

    def remove_rular(self, arg):
        #base = arg.split(".")[0]
        #img = cv2.cvtColor(arg, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arg, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = arg.shape
        imgArea = height*width

        # Get bounding box x,y,w,h of each contours
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: (x[2]*x[3]), reverse=True)
        # consider those rectangle whose area>10000 and less than one-fourth of images
        rects = [r for r in rects if (
            imgArea*self.parameter['maxRularArea']) > (r[2]*r[3]) > (imgArea*self.parameter['minRularArea'])]

        # detect child rectangles. Usually those are not ruler. Rular position are basically any one side.
        removeRect = []
        for i, rect1 in enumerate(rects):
            (x1, y1, w1, h1) = rect1
            for rect2 in rects[i+1:len(rects)]:
                (x2, y2, w2, h2) = rect2
                if (x1 < x2) and (y1 < y2) and (x1+w1 > x2+w2) and (y1+h1 > y2+h2):
                    removeRect.append(rect2)

        # removed child rectangles.
        rects = [x for x in rects if x not in removeRect]

        predictRular = []
        for rect in rects:
            (x, y, w, h) = rect
            if (w < width*self.parameter['rularWidth']) and ((y > height*self.parameter['positionBelow']) or ((x+w) < width*self.parameter['positionLeft']) or (x > width*self.parameter['positionRight'])):
                if (self.parameter['rularRatioMin'] < round(float(w)/float(h), 2) < self.parameter['rularRatioMax']) or (self.parameter['rularRatioMin'] < round(float(h)/float(w), 2) < self.parameter['rularRatioMax']):
                    blackPixel = np.count_nonzero(arg[y:y+h, x:x+w] == 0)
                    predictRular.append((x, y, w, h, blackPixel))

        # Finally check number of black pixel to avoid false rular
        if predictRular:
            predictRular = sorted(
                predictRular, key=lambda x: (x[4]), reverse=True)
            x, y, w, h, _ = predictRular[0]
            cv2.rectangle(arg, (x-15, y-15), (x+w+20, y+h+20),
                          (255, 255, 255), cv2.FILLED)

        return arg

    def BorderLine(self, MaxBoundary, lines, index, flag, lineDetectH, lineDetectV):
        getLine = 1
        LastLine = []
        if flag in ('top', 'left'):
            for i in range(len(lines)-1):
                if(abs(lines[i][index]-lines[i+1][index])) <= 15 and lines[i][index] < MaxBoundary:
                    LastLine = [lines[i][0], lines[i]
                                [1], lines[i][2], lines[i][3]]
                    getLine += 1
                elif getLine >= 3:
                    break
                else:
                    getLine = 1
        elif flag in ('bottom', 'right'):
            for i in reversed(list(range(len(lines)-1))):
                if(abs(lines[i][index]-lines[i+1][index])) <= 15 and lines[i][index] > MaxBoundary:
                    LastLine = [lines[i][0], lines[i]
                                [1], lines[i][2], lines[i][3]]
                    getLine += 1
                elif getLine >= 3:
                    break
                else:
                    getLine = 1
        if getLine >= 3 and LastLine:
            if flag == "top":
                lineDetectH.append((
                    LastLine[0], max(LastLine[1], LastLine[3]),
                    LastLine[2], max(LastLine[1], LastLine[3])
                ))
            if flag == "left":
                lineDetectV.append((
                    max(LastLine[0], LastLine[2]), LastLine[1],
                    max(LastLine[0], LastLine[2]), LastLine[3]
                ))
            if flag == "bottom":
                lineDetectH.append((
                    LastLine[0], min(LastLine[1], LastLine[3]),
                    LastLine[2], min(LastLine[1], LastLine[3])
                ))
            if flag == "right":
                lineDetectV.append((
                    min(LastLine[0], LastLine[2]), LastLine[1],
                    min(LastLine[0], LastLine[2]), LastLine[3]
                ))

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

    def detect_lines(self, arg):
        Hline = []
        Vline = []
        gray = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        imgHeight, imgWidth, _ = arg.shape
        lines = lsd(gray)

        for i in range(lines.shape[0]):
            pt1 = (int(lines[i, 0]), int(lines[i, 1]))
            pt2 = (int(lines[i, 2]), int(lines[i, 3]))
            # consider those line whise length more than this orbitrary value
            if (abs(pt1[0]-pt2[0]) > 45) and ((int(pt1[1]) < imgHeight*0.25) or (int(pt1[1]) > imgHeight*0.75)):
                # make full horizontal line
                Hline.append([0, int(pt1[1]), imgWidth, int(pt2[1])])
            if (abs(pt1[1]-pt2[1]) > 45) and ((int(pt1[0]) < imgWidth*0.4) or (int(pt1[0]) > imgWidth*0.6)):
                # make full vertical line
                Vline.append([int(pt1[0]), 0, int(pt2[0]), imgHeight])
        Hline.sort(key=lambda x: (x[1]), reverse=False)
        Vline.sort(key=lambda x: (x[0]), reverse=False)
        return imgHeight, imgWidth, Hline, Vline

    def select_borderLine(self, arg, lineDetectH, lineDetectV):
        imgHeight, imgWidth, Hlines, Vlines = self.detect_lines(arg)

        # top side
        self.BorderLine(imgHeight*0.25, Hlines, 1,
                        "top", lineDetectH, lineDetectV)
        # left side
        self.BorderLine(imgWidth*0.4, Vlines, 0, "left",
                        lineDetectH, lineDetectV)
        # bottom side
        self.BorderLine(imgHeight*0.75, Hlines, 1,
                        "bottom", lineDetectH, lineDetectV)
        # right side
        self.BorderLine(imgWidth*0.6, Vlines, 0, "right",
                        lineDetectH, lineDetectV)

        intersectPoint = []
        for l1 in lineDetectH:
            for l2 in lineDetectV:
                x, y = self.get_intersect(
                    (l1[0], l1[1]),
                    (l1[2], l1[3]),
                    (l2[0], l2[1]),
                    (l2[2], l2[3])
                )
                intersectPoint.append([x, y])
        Xstart = 0
        Xend = imgWidth
        Ystart = 0
        Yend = imgHeight
        for i in intersectPoint:
            Xs = int(i[0])+10 if i[0] < imgWidth*0.4 else 10
            if Xs > Xstart:
                Xstart = Xs
            Xe = int(i[0])-10 if i[0] > imgWidth*0.6 else int(imgWidth)-10
            if Xe < Xend:
                Xend = Xe
            Ys = int(i[1])+10 if i[1] < imgHeight*0.25 else 10
            # print("Ys,Ystart:",Ys,Ystart)
            if Ys > Ystart:
                Ystart = Ys
            Ye = int(i[1])-15 if i[1] > imgHeight*0.75 else int(imgHeight)-15
            if Ye < Yend:
                Yend = Ye

        if Xend < 0:
            Xend = 10
        if Yend < 0:
            Yend = 15
        #self.save_pf(base, [Xstart, Ystart, Xend, Yend])

        return [Xstart, Ystart, Xend, Yend]

    def filter_noisebox(self, textarea, height, width):
        tmp = []
        st = True

        while st:
            textarea = [list(x) for x in textarea if x not in tmp]
            if len(textarea) > 1:
                tmp = []
                textarea = sorted(
                    textarea, key=lambda x: (x[3]), reverse=False)
                # print textarea
                x11, y11, x12, y12 = textarea[0]
                x21, y21, x22, y22 = textarea[1]

                if abs(y12-y21) > 100 and (float(abs(x12-x11)*abs(y12-y11))/(height*width)) < 0.001:
                    tmp.append(textarea[0])

                x11, y11, x12, y12 = textarea[-2]
                x21, y21, x22, y22 = textarea[-1]

                if abs(y12-y21) > 100 and (float(abs(x21-x22)*abs(y22-y21))/(height*width)) < 0.001:
                    tmp.append(textarea[-1])

                if len(tmp) == 0:
                    st = False
            else:
                break

        return textarea

    def detect_textarea(self, arg):
        textarea = []
        small = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        height, width, _ = arg.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

        _, bw = cv2.threshold(
            grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (10, 1))  # for historical docs
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)

        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            # print x,y,w,h
            mask[y:y+h, x:x+w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

            if r > 0.45 and (width*0.9) > w > 15 and (height*0.5) > h > 15:
                textarea.append([x, y, x+w-1, y+h-1])
                cv2.rectangle(arg, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)

        if len(textarea) > 1:
            textarea = self.filter_noisebox(textarea, height, width)

        return textarea, arg, height, width

    def save_pf(self, base, textarea):
        x1, y1, x2, y2 = textarea

        img = Image.open(base+'.pf.png')
        img2 = img.crop((x1, y1, x2, y2))
        img2.save(base + '.pf.png')
        self.write_crop_coordinate(base, textarea)

    def filter_area(self, textarea, binImg):
        height, width, _ = binImg.shape
        tmp = []
        for area in textarea:
            if (height*width*self.parameter['minArea'] < (abs(area[2]-area[0]) * abs(area[3]-area[1]))):
                tmp.append(area)
        return tmp

    def marge_columns(self, textarea, colSeparator):
        tmp = []
        marge = []
        #  height, _ = binImg.shape
        textarea.sort(key=lambda x: (x[0]))
        for i in range(len(textarea)-1):
            st = False
            x11, y11, x12, y12 = textarea[i]
            x21, y21, x22, y22 = textarea[i+1]
            if x21-x12 <= colSeparator:
                if len(marge) > 0:
                    # print "marge ", marge[0]
                    x31, y31, x32, y32 = marge[0]
                    marge.pop(0)
                else:
                    x31, y31, x32, y32 = [9999, 9999, 0, 0]
                marge.append([min(x11, x21, x31), min(y11, y21, y31),
                              max(x12, x22, x32), max(y12, y22, y32)])
                st = True
            else:
                tmp.append(textarea[i])

        if not st:
            tmp.append(textarea[-1])

        return tmp+marge

    def crop_area(self, textarea, binImg, rgb, colSeparator):
        height, width, _ = binImg.shape

        textarea = np.unique(textarea, axis=0)
        i = 0
        tmp = []
        areas = []
        while i < len(textarea):
            textarea = [list(x) for x in textarea if x not in tmp]
            tmp = []
            if len(textarea) == 0:
                break
            maxBox = textarea[0]
            for chkBox in textarea:
                if maxBox != chkBox:
                    x11, y11, x12, y12 = maxBox
                    x21, y21, x22, y22 = chkBox
                    if ((x11 <= x21 <= x12) or (x21 <= x11 <= x22)):
                        tmp.append(maxBox)
                        tmp.append(chkBox)
                        maxBox = [min(x11, x21), min(y11, y21),
                                  max(x12, x22), max(y12, y22)]
            if len(tmp) == 0:
                tmp.append(maxBox)
            x1, y1, x2, y2 = maxBox
            areas.append(maxBox)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            i = i+1

        textarea = np.unique(areas, axis=0).tolist()
        if len(textarea) > 0:
            textarea = self.filter_area(textarea, binImg)
        if len(textarea) > 1:
            textarea = self.marge_columns(textarea, colSeparator)
            # print textarea

        if len(textarea) > 0:
            textarea = sorted(textarea, key=lambda x: (
                (x[2]-x[0])*(x[3]-x[1])), reverse=True)
            # print textarea
            x1, y1, x2, y2 = textarea[0]
            x1 = x1-20 if x1 > 20 else 0
            x2 = x2+20 if x2 < width-20 else width
            y1 = y1-40 if y1 > 40 else 0
            y2 = y2+40 if y2 < height-40 else height

            #self.save_pf(base, [x1, y1, x2, y2])

        return textarea

    def process(self):
        """Performs border detection on the workspace. """
        try:
            LOG.info("OUTPUT FILE %s", self.output_file_grp)
            page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info(
                "No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
        oplevel = self.parameter['operation_level']
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID

            LOG.info("INPUT FILE %i / %s", n, page_id)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            # Check for existing Border --> already cropped
            border = page.get_Border()
            if border:
                left, top, right, bottom = bbox_from_points(
                    border.get_Coords().points)
                LOG.warning('Overwriting existing Border: %i:%i,%i:%i',
                            left, top, right, bottom)

            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 Labels=[LabelsType(  # externalRef="parameters",
                                         Label=[LabelType(type_=name,
                                                          value=self.parameter[name])
                                                for name in self.parameter.keys()])]))
            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id,
                feature_filter='cropped',
                feature_selector='binarized') # should also be deskewed

            if oplevel == "page":
                self._process_segment(
                    page_image, page, page_coords, page_id, input_file, n)
            else:
                raise Exception(
                    'Operation level %s, but should be "page".', oplevel)
            file_id = input_file.ID.replace(
                self.input_file_grp, page_grp)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            if file_id == input_file.ID:
                file_id = concat_padded(page_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(page_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts).encode('utf-8')
            )

    def _process_segment(self, page_image, page, page_xywh, page_id, input_file, n):
        img_array = ocrolib.pil2array(page_image)

        # Check if image is RGB or not #FIXME: check not needed anymore?
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)

        img_array_bin = np.array(
            img_array > ocrolib.midrange(img_array), 'i')

        lineDetectH = []
        lineDetectV = []
        img_array_rr = self.remove_rular(img_array)

        textarea, img_array_rr_ta, height, width = self.detect_textarea(
            img_array_rr)
        colSeparator = int(
            width * self.parameter['colSeparator'])
        if len(textarea) > 1:
            textarea = self.crop_area(
                textarea, img_array_bin, img_array_rr_ta, colSeparator)

            if len(textarea) == 0:
                min_x, min_y, max_x, max_y = self.select_borderLine(
                    img_array_rr, lineDetectH, lineDetectV)
            else:
                min_x, min_y, max_x, max_y = textarea[0]
        elif len(textarea) == 1 and (height*width*0.5 < (abs(textarea[0][2]-textarea[0][0]) * abs(textarea[0][3]-textarea[0][1]))):
            x1, y1, x2, y2 = textarea[0]
            x1 = x1-20 if x1 > 20 else 0
            x2 = x2+20 if x2 < width-20 else width
            y1 = y1-40 if y1 > 40 else 0
            y2 = y2+40 if y2 < height-40 else height

            min_x, min_y, max_x, max_y = textarea[0]
        else:
            min_x, min_y, max_x, max_y = self.select_borderLine(
                img_array_rr, lineDetectH, lineDetectV)

        border_polygon = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        border_polygon = coordinates_for_segment(border_polygon, page_image, page_xywh)
        border_points = points_from_polygon(border_polygon)
        brd = BorderType(Coords=CoordsType(border_points))
        page.set_Border(brd)

        page_image = crop_image(page_image, box=(min_x, min_y, max_x, max_y))
        page_xywh['features'] += ',cropped'

        file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
        if file_id == input_file.ID:
            file_id = concat_padded(self.image_grp, n)

        file_path = self.workspace.save_image_file(page_image,
                                                   file_id,
                                                   page_id=page_id,
                                                   file_grp=self.image_grp)
        page.add_AlternativeImage(AlternativeImageType(
            filename=file_path, comments=page_xywh['features']))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrCropper, *args, **kwargs)
