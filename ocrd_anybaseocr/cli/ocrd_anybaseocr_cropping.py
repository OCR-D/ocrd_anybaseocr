# pylint: disable=invalid-name
# A sample image and METS (repo/assets/data/dfki-testdata/data/mets.xml,
# page becker_quaestio_1586_00013) is provided, see make test.

# Workflow should be: binarization, deskewing, cropping and dewarping
# (but can also be: binarization, dewarping, deskewing, and cropping).

# Takes a document image as input and crops/selects the page frame
# (text content area), removing textual noise as well as any other
# noise near the margins (including a ruler placed in the photo).

# Copyright 2018 Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Apache License 2.0

# A permissive license whose main conditions require preservation of copyright
# and license notices. Contributors provide an express grant of patent rights.
# Licensed works, modifications, and larger works may be distributed under
# different terms and without source code.

import os
from types import SimpleNamespace
import numpy as np
from pylsd.lsd import lsd
from shapely.geometry import Polygon
import cv2
from PIL import Image
from scipy.spatial import distance_matrix
from scipy.stats import linregress

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
DEBUG_INTERACTIVE = False
if DEBUG:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch
    from tempfile import mkstemp
    def dshow(title):
        if DEBUG_INTERACTIVE:
            plt.legend(handles=[Patch(label=title)])
            plt.show() # blocks
        else:
            fd, fname = mkstemp(suffix='_' + title.replace(' ', '_') + '.png')
            plt.savefig(fname, dpi=600)
            os.close(fd)
            plt.clf()

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
        self.logger = None

    def detect_ruler(self, arg):
        gray = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = arg.shape
        imgArea = height * width
        minArea = imgArea * self.parameter['rulerAreaMin']
        maxArea = imgArea * self.parameter['rulerAreaMax']

        # Get bounding box x,y,w,h of each contours
        rects = [cv2.boundingRect(cnt) for cnt in contours]
        rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
        # consider those rectangle whose area>10000 and less than one-fourth of images
        rects = [rect for rect in rects if maxArea > rect[2] * rect[3] > minArea]

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
                    blackPixel = np.count_nonzero(bw[y:y+h, x:x+w])
                    predictRuler.append((x, y, w, h, blackPixel))
                    self.logger.debug("detected ruler candidate: %dx%d+%dx%d (%dpx fg)",
                                      x, y, w, y, blackPixel)

        # Finally check number of black pixel to avoid false ruler
        if predictRuler:
            # sort by fg size
            predictRuler = sorted(predictRuler, key=lambda x: x[4], reverse=True)
            # pick largest candidate
            x, y, w, h, a = predictRuler[0]
            self.logger.info("detected ruler at %dx%d+%dx%d", x, y, w, h)
            # create mask image
            mask = np.zeros(bw.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x-15, y-15), (x+w+20, y+h+20), 255, cv2.FILLED)
            if DEBUG:
                plt.imshow(mask)
                dshow('ruler mask')
            return mask, (x, y, w, h)

        return None, None

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
        lines = lsd(gray, ang_th=60, sigma_scale=3.0)
        if DEBUG:
            plt.imshow(arg)
            for x1, y1, x2, y2, _ in lines:
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=2, linestyle='dashed'))
            dshow('line segments')
        imgHeight, imgWidth, _ = arg.shape
        y1max = self.parameter['marginTop'] * imgHeight
        y2min = self.parameter['marginBottom'] * imgHeight
        x1max = self.parameter['marginLeft'] * imgWidth
        x2min = self.parameter['marginRight'] * imgWidth
        hlines = []
        vlines = []
        for x1, y1, x2, y2, w in lines:
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            # consider line segments near margins and not too short
            if dx > 15 and dy / dx < 0.15 and (y1 < y1max or y2 > y2min):
                hlines.append([x1, y1, x2, y2, w])
            if dy > 15 and dx / dy < 0.15 and (x1 < x1max or x2 > x2min):
                vlines.append([x1, y1, x2, y2, w])

        return imgHeight, imgWidth, hlines, vlines

    def aggregate_lines(self, arg, lines, is_vertical,
                        min_length=150, # minimum total length of line group
                        min_end=None, # start of allowed range (straight axis)
                        max_start=None, # end of allowed range (straight axis)
                        min_pos=None, # start of forbidden range (perpendicular axis)
                        max_pos=None): # end of forbidden range (perpendicular axis)
        imgHeight, imgWidth, _ = arg.shape
        if not lines:
            return []
        lines = np.array(lines)
        if is_vertical:
            rng = 1
            if not min_end:
                min_end = lines[:,3].max()
            if not max_start:
                max_start = lines[:,1].min()
            if not min_pos:
                min_pos = lines[:,[0,2]].max()
            if not max_pos:
                max_pos = lines[:,[0,2]].min()
        else:
            rng = 0
            if not min_end:
                min_end = lines[:,2].max()
            if not max_start:
                max_start = lines[:,0].min()
            if not min_pos:
                min_pos = lines[:,[1,3]].max()
            if not max_pos:
                max_pos = lines[:,[1,3]].min()
        # result: line lengths on diagonal, distances on non-diagonals
        dist = distance_matrix(lines[:, 0:2], lines[:, 2:4])
        # find clustering of line segments by distance and direction
        groups = list() # list of Group
        class Group(SimpleNamespace):
            # ind: hlines/vlines index set
            # res: linear x-y/y-x regressor
            pass
        # initialize single-line groups
        for i, line in enumerate(lines):
            points = np.vstack([lines[i, 0:2], lines[i, 2:4]])
            if is_vertical:
                newres = linregress(points[:, 1], points[:, 0])
            else:
                newres = linregress(points[:, 0], points[:, 1])
            groups.append(Group(ind={i}, res=newres, wgt=lines[i, 4]))
        # merge nearby groups (with mutual points of small distance)
        for start, end in zip(*np.unravel_index(np.argsort(dist, None), dist.shape)):
            if start == end:
                continue # ignore diagonals (line length)
            if dist[start, end] > 15:
                break # ignore points too far apart
            for i, igroup in enumerate(groups):
                if not igroup.res: continue # already merged / to be deleted
                if (start in igroup.ind) == (end in igroup.ind):
                    continue
                for j, jgroup in enumerate(groups[i+1:], i+1):
                    if not jgroup.res: continue # already merged / to be deleted
                    if (start if end in igroup.ind else end) not in jgroup.ind:
                        continue
                    newind = igroup.ind.union(jgroup.ind)
                    ind = np.array(list(newind))
                    points = np.concatenate([lines[ind, 0:2], lines[ind, 2:4]])
                    if is_vertical:
                        newres = linregress(points[:, 1], points[:, 0])
                    else:
                        newres = linregress(points[:, 0], points[:, 1])
                    if (newres.stderr > 0.04 or
                        newres.stderr - igroup.res.stderr > 0.02 or
                        newres.stderr - jgroup.res.stderr > 0.02):
                        continue # ignore line segments deviating in direction too much
                    #print("merging {} and {}".format(igroup.ind, jgroup.ind))
                    iind = np.array(list(igroup.ind))
                    ilength = dist[iind,iind].sum()
                    jind = np.array(list(jgroup.ind))
                    jlength = dist[jind,jind].sum()
                    igroup.wgt = (ilength * igroup.wgt + jlength * jgroup.wgt) / (ilength + jlength)
                    igroup.ind = newind
                    igroup.res = newres
                    jgroup.res = None # mark for deletion
        # merge similar groups (with approximately the same direction and no gaps)
        for i, igroup in enumerate(groups):
            if not igroup.res: continue # already merged / to be deleted
            for j, jgroup in enumerate(groups[i+1:], i+1):
                if not jgroup.res: continue # already merged / to be deleted
                if not (abs(igroup.res.intercept - jgroup.res.intercept) < 0.01 * imgWidth and
                        abs(igroup.res.intercept - jgroup.res.intercept + \
                            (igroup.res.slope - jgroup.res.slope) * imgWidth) < 0.01 * imgWidth):
                    # inconsistent directions
                    # self.logger.debug("%s/%s:\ninconsistent %s/%s",
                    #                   str(igroup.ind), str(jgroup.ind),
                    #                   str(igroup.res), str(jgroup.res))
                    continue
                iind = np.array(list(igroup.ind))
                jind = np.array(list(jgroup.ind))
                ipoints = np.concatenate([lines[iind, 0:2], lines[iind, 2:4]])
                jpoints = np.concatenate([lines[jind, 0:2], lines[jind, 2:4]])
                istart, iend = ipoints[:,rng].min(), ipoints[:,rng].max()
                jstart, jend = jpoints[:,rng].min(), jpoints[:,rng].max()
                if not (0 < jstart - iend < 0.1 * imgWidth or
                        0 < istart - jend < 0.1 * imgWidth):
                    # too large gap
                    # self.logger.debug("%s/%s:\nlarge gap %d:%d/%d:%d",
                    #                   str(igroup.ind), str(jgroup.ind),
                    #                   istart, iend, jstart, jend)
                    continue
                newind = igroup.ind.union(jgroup.ind)
                points = np.concatenate([ipoints, jpoints])
                if is_vertical:
                    newres = linregress(points[:, 1], points[:, 0])
                else:
                    newres = linregress(points[:, 0], points[:, 1])
                if (newres.stderr > 0.04 or
                    newres.stderr - igroup.res.stderr > 0.02 or
                    newres.stderr - jgroup.res.stderr > 0.02):
                    # self.logger.debug("%s/%s:\nnoisy %f over %f/%f",
                    #                   str(igroup.ind), str(jgroup.ind),
                    #                   newres.stderr, igroup.res.stderr, jgroup.res.stderr)
                    continue # ignore line segments deviating in direction too much
                #print("merging {} and {}".format(igroup.ind, jgroup.ind))
                iind = np.array(list(igroup.ind))
                ilength = dist[iind,iind].sum()
                jind = np.array(list(jgroup.ind))
                jlength = dist[jind,jind].sum()
                igroup.wgt = (ilength * igroup.wgt + jlength * jgroup.wgt) / (ilength + jlength)
                igroup.ind = newind
                igroup.res = newres
                jgroup.res = None # mark for deletion
        if DEBUG:
            plt.imshow(arg)
            for group in groups:
                if not group.res:
                    continue
                ind = np.array(list(group.ind))
                points = np.concatenate([lines[ind,rng], lines[ind,2+rng]])
                if is_vertical:
                    # x = slope * y + intercept
                    y1 = points.min()
                    x1 = group.res.slope * y1 + group.res.intercept
                    y2 = points.max()
                    x2 = group.res.slope * y2 + group.res.intercept
                else:
                    # y = slope * x + intercept
                    x1 = points.min()
                    y1 = group.res.slope * x1 + group.res.intercept
                    x2 = points.max()
                    y2 = group.res.slope * x2 + group.res.intercept
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=max(1,int(group.wgt/2)), linestyle='dashed'))
                #plt.gca().text(x1, y1, str(group.ind), bbox={'clip_box': [[x1,y1],[x2,y2]]})
            dshow('line groups')
        self.logger.debug("detected %d %s line groups", len(groups),
                          "vertical" if is_vertical else "horizontal")
        for group in groups.copy():
            if not group.res:
                # merged already
                groups.remove(group)
                continue
            ind = np.array(list(group.ind))
            lengths = dist[ind,ind]
            group.length = lengths.sum()
            if group.length < min_length:
                # total length of lines in group is too short
                groups.remove(group)
                #print("too short: {} ({} < {})".format(group.ind, group.length, min_length))
                continue
            points = np.concatenate([lines[ind,rng], lines[ind,2+rng]])
            group.start = points.min()
            group.end = points.max()
            if group.start > max_start or group.end < min_end:
                # lines in group are completely in margin areas
                groups.remove(group)
                if group.start > max_start:
                    pass #print("invalid range: {} ({} > {})".format(group.ind, group.start, max_start))
                else:
                    pass #print("invalid range: {} ({} < {})".format(group.ind, group.end, min_end))
                continue
            if is_vertical:
                # x = slope * y + intercept
                y1 = 0
                y2 = imgHeight
                x1 = group.res.intercept
                x2 = group.res.slope * y2 + x1
                group.pos = 0.5 * (x1 + x2)
                if min_pos < group.pos < max_pos:
                    # lines in group are completely in margin areas
                    groups.remove(group)
                    #print("invalid position: {} ({} < {} < {})".format(group.ind, min_pos, 0.5 * (x1+x2), max_pos))
                    continue
            else:
                # y = slope * x + intercept
                x1 = 0
                x2 = imgWidth
                y1 = group.res.intercept
                y2 = group.res.slope * x2 + y1
                group.pos = 0.5 * (y1 + y2)
                if min_pos < group.pos < max_pos:
                    # lines in group are completely in margin areas
                    groups.remove(group)
                    #print("invalid position: {} ({} < {} < {})".format(group.ind, min_pos, 0.5 * (y1+y2), max_pos))
                    continue
            group.line = [x1, y1, x2, y2]
            #print("kept {}".format(group.ind))
        self.logger.debug("keeping %d line candidates after filtering", len(groups))
        if DEBUG:
            plt.imshow(arg)
            for group in groups:
                x1, y1, x2, y2 = group.line
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=1, linestyle='-', color='red'))
            dshow('line candidates')
        return groups

    def select_borderLine(self, arg, mask=None):
        imgHeight, imgWidth, Hlines, Vlines = self.detect_lines(arg)
        perfect = True
        y1max = self.parameter['marginTop'] * imgHeight
        y2min = self.parameter['marginBottom'] * imgHeight
        x1max = self.parameter['marginLeft'] * imgWidth
        x2min = self.parameter['marginRight'] * imgWidth
        # connect consistent line segments and filter by position and length
        Hgroups = self.aggregate_lines(arg, Hlines, False, 0.2 * imgWidth, x1max, x2min, y1max, y2min)
        Vgroups = self.aggregate_lines(arg, Vlines, True, 0.2 * imgHeight, y1max, y2min, x1max, x2min)
        # split horizontal lines into top and bottom candidates,
        # split vertical lines into left and right candidates:
        toplines = filter(lambda group: # y pos at top margin
                          group.pos < y1max, Hgroups)
        botlines = filter(lambda group: # y pos at bottom margin
                          group.pos > y2min, Hgroups)
        lftlines = filter(lambda group: # x pos at left margin
                          group.pos < x1max, Vgroups)
        rgtlines = filter(lambda group: # x pos at right margin
                          group.pos > x2min, Vgroups)
        if mask is not None:
            # apply outer boundaries where ruler is:
            mask_x, mask_y, mask_w, mask_h = mask
            x2maxDist = mask_x - 0.5 * imgWidth
            y2maxDist = mask_y - 0.5 * imgHeight
            x1minDist = 0.5 * imgWidth - (mask_x + mask_w)
            y1minDist = 0.5 * imgHeight - (mask_y + mask_h)
            bestDist = max(x2maxDist, y2maxDist, x1minDist, y1minDist)
            if bestDist == y1minDist:
                toplines = filter(lambda group: # y pos below ruler
                                  group.pos > mask_y + mask_h, toplines)
            elif bestDist == y2maxDist:
                botlines = filter(lambda group: # y pos above ruler
                                  group.pos < mask_y, botlines)
            elif bestDist == x1minDist:
                lftlines = filter(lambda group: # x pos right of ruler
                                  group.pos > mask_x + mask_w, lftlines)
            elif bestDist == x2maxDist:
                rgtlines = filter(lambda group: # x pos left of ruler
                                  group.pos < mask_x, rgtlines)
        # select best (i.e. innermost longest) candidate (or fallback) on each side
        def attenuate_pos(x):
            x0 = 3 * x # peak at 30% max-margin (i.e. disprefer smaller and larger than that position)
            return x0 / np.exp(x0)
        toplines = sorted(toplines, reverse=True,
                          key=lambda group: # maximize product of length and y pos
                          group.wgt**2 * group.length * attenuate_pos(group.pos / y1max))
        if toplines:
            self.logger.info("found top margin (pos: %d, length: %d)",
                             toplines[0].pos, toplines[0].length)
            topline = toplines[0].line
        else:
            perfect = False
            topline = [0, 0, imgWidth, 0]
        botlines = sorted(botlines, reverse=True,
                          key=lambda group: # maximize product of length and h-y pos
                          group.wgt**2 * group.length * attenuate_pos((imgHeight - group.pos) / (imgHeight - y2min)))
        if botlines:
            self.logger.info("found bottom margin (pos: %d, length: %d)",
                             botlines[0].pos, botlines[0].length)
            botline = botlines[0].line
        else:
            perfect = False
            botline = [0, imgHeight, imgWidth, imgHeight]
        lftlines = sorted(lftlines, reverse=True,
                          key=lambda group: # maximize product of length and x pos
                          group.wgt**2 * group.length * attenuate_pos(group.pos / x1max))
        if lftlines:
            self.logger.info("found left margin (pos: %d, length: %d)",
                             lftlines[0].pos, lftlines[0].length)
            lftline = lftlines[0].line
        else:
            perfect = False
            lftline = [0, 0, 0, imgHeight]
        rgtlines = sorted(rgtlines, reverse=True,
                          key=lambda group: # maximize product of length and w-x pos
                          group.wgt**2 * group.length * attenuate_pos((imgWidth - group.pos) / (imgWidth - x2min)))
        if rgtlines:
            self.logger.info("found right margin (pos: %d, length: %d)",
                             rgtlines[0].pos, rgtlines[0].length)
            rgtline = rgtlines[0].line
        else:
            rgtline = [imgWidth, 0, imgWidth, imgHeight]
            perfect = False
        if DEBUG:
            plt.imshow(arg)
            for x1, y1, x2, y2 in [topline, botline, lftline, rgtline]:
                plt.gca().add_artist(Line2D((x1,x2), (y1,y2), linewidth=2, linestyle='dotted'))
            dshow('border lines')
        # intersect all sides
        intersectPoint = []
        for hx1, hy1, hx2, hy2 in [topline, botline]:
            for vx1, vy1, vx2, vy2 in [lftline, rgtline]:
                x, y = self.get_intersect((hx1, hy1),
                                          (hx2, hy2),
                                          (vx1, vy1),
                                          (vx2, vy2))
                intersectPoint.append([x, y])
            lftline, rgtline = rgtline, lftline
        # FIXME: return confidence value (length and no fallback on each side)
        return intersectPoint, perfect

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

    def detect_textboxes(self, arg, mask=None):
        textboxes = []
        gray = cv2.cvtColor(arg, cv2.COLOR_RGB2GRAY)
        height, width, _ = arg.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        if DEBUG:
            plt.imshow(grad)
            dshow('morphological gradient')

        _, bw = cv2.threshold(
            grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if isinstance(mask, np.ndarray):
            bw[mask > 0] = 0
        if DEBUG:
            plt.imshow(bw)
            dshow('binarized gradient')

        # for x1, y1, x2, y2, w in lines:
        #     l = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        #     if l > 40 and l / w > 20:
        #         print("suppressing %d,%d %d,%d (%d)" %(
        #             x1, y1, x2, y2, w))
        #         cv2.line(bw, (int(x1),int(y1)), (int(x2),int(y2)), 0, int(np.ceil(w / 2)))
        # if DEBUG:
        #     plt.imshow(bw)
        #     dshow('without lines')
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (10, 1))  # for historical docs
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        if DEBUG:
            plt.imshow(connected)
            dshow('horizontal closing')

        contours, hierarchy = cv2.findContours(
            connected.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if DEBUG:
            mask = np.zeros(bw.shape[:2], dtype=np.uint16)
            res = arg.copy()

        def apply_contour(idx):
            while 0 <= idx < len(contours):
                next_idx, prev_idx, child_idx, parent_idx = hierarchy[0, idx].tolist()
                if len(contours[idx]) >= 3:
                    x, y, w, h = cv2.boundingRect(contours[idx])
                    if DEBUG: cv2.drawContours(mask, contours, idx, idx+1, -1)
                    r = cv2.contourArea(contours[idx]) / (w * h)
                    if r > 0.25 and (width*0.9) > w > 15 and (height*0.5) > h > 15:
                        textboxes.append([x, y, x+w-1, y+h-1])
                        if DEBUG: cv2.rectangle(res, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
                    if child_idx >= 0:
                        apply_contour(child_idx)
                idx = next_idx
        if contours:
            apply_contour(0)
        self.logger.debug("found %d raw text boxes", len(textboxes))
        if DEBUG:
            mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            plt.imshow(mask)
            dshow('contours')

        if len(textboxes) > 1:
            textboxes = self.filter_noisebox(textboxes, height, width)
        if DEBUG:
            plt.imshow(res)
            dshow('text boxes')

        return textboxes

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

    def merge_boxes(self, textboxes, img):
        height, width, _ = img.shape
        y1max = self.parameter['marginTop'] * height
        y2min = self.parameter['marginBottom'] * height
        x1max = self.parameter['marginLeft'] * width
        x2min = self.parameter['marginRight'] * width
        colSeparator = int(width * self.parameter['columnSepWidthMax'])

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
        self.logger.debug("merged into %d text boxes (i.e. columns)", len(boxes))
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in boxes:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            dshow('columns')

        columns = np.unique(boxes, axis=0).tolist()
        # remove margin-only results
        def nonmargin(box):
            x1, y1, x2, y2 = box
            return not (x2 < x1max or
                        y2 < y1max or
                        x1 > x2min or
                        y1 > y2min)
        columns = list(filter(nonmargin, columns))
        self.logger.debug("filtered into %d columns by margins", len(columns))
        # merge columns
        if len(columns) > 1:
            columns = self.merge_columns(columns, colSeparator)
            self.logger.debug("merged into %d columns by sepwidth", len(columns))
        # filter by size
        if len(columns) > 0:
            minArea = height * width * self.parameter['columnAreaMin']
            columns = list(filter(lambda box: self.get_area(box) > minArea, columns))
        self.logger.debug("filtered into %d columns by area", len(columns))
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in columns:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            dshow('minArea columns')

        if len(columns) > 0:
            columns = sorted(columns, key=self.get_area, reverse=True)
        if DEBUG:
            plt.imshow(img)
            for x1, y1, x2, y2 in columns:
                plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                              alpha=0.7, linewidth=2, linestyle='dashed'))
            dshow('merged columns')

        return columns

    def process(self):
        """Performs heuristic page frame detection (cropping) on the workspace.
        
        Open and deserialize PAGE input files and their respective images.
        (Input should be deskewed already.) Retrieve the raw (non-binarized,
        uncropped) page image.
        
        Detect line segments via edge gradients, and cluster them into contiguous
        horizontal and vertical lines if possible. If candidates which are located
        at the margin and long enough (covering a large fraction of the page) exist
        on all four sides, then pick the best (i.e. thickest, longest and inner-most)
        one on each side and use their intersections as border points.
        
        Otherwise, first try to detect a ruler (i.e. image segment depicting a rule
        placed on the scan/photo for scale references) via thresholding and contour
        detection, identifying a single large rectangular region with a certain aspect
        ratio. Suppress (mask) any such segment during further calculations.

        Next in that line, try to detect text segments on the page. For that purpose,
        get the gradient of grayscale image, threshold and morphologically close it,
        then determine contours to define approximate text boxes. Merge these into
        columns, filtering candidates too small or entirely in the margin areas.
        Finally, merge the remaining columns across short gaps. If only one column
        remains, and it covers a significant fraction of the page, pick that segment
        as solution.
        
        Otherwise, keep the border points derived from line segments (intersecting
        with the full image on each side without line candidates).
        
        Lastly, map coordinates to the original (undeskewed) image and intersect
        the border polygon with the full image frame. Use that to define the page's
        Border.
        
        Moreover, crop (and mask) the image accordingly, and reference the
        resulting image file as AlternativeImage in the Page element.
        Add the new image file to the workspace along with the output fileGrp,
        and using a file ID with suffix ``.IMG-CROP`` along with further
        identification of the input element.
        
        Produce new output files by serialising the resulting hierarchy.
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        self.logger = getLogger('processor.AnybaseocrCropper')

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            self.logger.info("INPUT FILE %i / %s", n, page_id)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            # Check for existing Border --> already cropped
            border = page.get_Border()
            if border:
                left, top, right, bottom = bbox_from_points(
                    border.get_Coords().points)
                self.logger.warning('Overwriting existing Border: %i:%i,%i:%i',
                                    left, top, right, bottom)

            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, # should be deskewed already
                feature_filter='cropped,binarized,grayscale_normalized')
            if self.parameter['dpi'] > 0:
                zoom = 300.0/self.parameter['dpi']
            elif page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi *= 2.54
                self.logger.info('Page "%s" uses %f DPI', page_id, dpi)
                zoom = 300.0/dpi
            else:
                zoom = 1

            self._process_page(page, page_image, page_coords, input_file, zoom)
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

    def _process_page(self, page, page_image, page_xywh, input_file, zoom=1.0):
        padding = self.parameter['padding']
        img_array = pil2array(page_image)
        # ensure RGB image
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        height, width, _ = img_array.shape
        size = height * width
        # zoom to 300 DPI (larger density: faster; most fixed parameters here expect 300)
        if zoom != 1.0:
            self.logger.info("scaling %dx%d image by %.2f", width, height, zoom)
            img_array = cv2.resize(img_array, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)

        # detect rule placed in image next to page for scale reference:
        mask_array, mask_box = self.detect_ruler(img_array)
        # detect page frame via line segment detector:
        border_polygon, prefer_border = self.select_borderLine(img_array, mask_box)
        border_polygon = np.array(border_polygon) / zoom # unzoom
        # pad inwards:
        border_polygon = Polygon(border_polygon).buffer(-padding).exterior.coords[:-1]
        # get the bounding box from the border polygon:
        # min_x, min_y = border_polygon.min(axis=0)
        # max_x, max_y = border_polygon.max(axis=0)
        # get the inner rectangle from the border polygon:
        # _, min_x, max_x, _ = np.sort(border_polygon[:,0])
        # _, min_y, max_y, _ = np.sort(border_polygon[:,1])
        if prefer_border:
            self.logger.info("Preferring line detector")
        else:
            self.logger.info("Falling back to text detector")
            textboxes = self.detect_textboxes(img_array, mask_array)
            if len(textboxes) > 1:
                textboxes = self.merge_boxes(textboxes, img_array)
            textboxes = np.array(textboxes) / zoom # unzoom

            if (len(textboxes) == 1 and
                self.parameter['columnAreaMin'] * size < self.get_area(textboxes[0])):
                self.logger.info("Using text area (%d%% area)",
                                 100 * self.get_area(textboxes[0]) / size)
                min_x, min_y, max_x, max_y = textboxes[0]
                # pad outwards
                border_polygon = polygon_from_bbox(min_x - padding,
                                                   min_y - padding,
                                                   max_x + padding,
                                                   max_y + padding)

        def clip(point):
            x, y = point
            x = max(0, min(page_image.width, x))
            y = max(0, min(page_image.height, y))
            return x, y
        border_polygon = coordinates_for_segment(border_polygon, page_image, page_xywh)
        border_polygon = list(map(clip, border_polygon))
        border_points = points_from_polygon(border_polygon)
        border = BorderType(Coords=CoordsType(border_points))
        page.set_Border(border)
        # get clipped relative coordinates for current image
        page_image, page_xywh, _ = self.workspace.image_from_page(
            page, input_file.pageId,
            fill='background', transparency=True)
        file_id = make_file_id(input_file, self.output_file_grp)
        file_path = self.workspace.save_image_file(page_image,
                                                   file_id + '.IMG-CROP',
                                                   page_id=input_file.pageId,
                                                   file_grp=self.output_file_grp)
        page.add_AlternativeImage(AlternativeImageType(
            filename=file_path, comments=page_xywh['features']))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrCropper, *args, **kwargs)
