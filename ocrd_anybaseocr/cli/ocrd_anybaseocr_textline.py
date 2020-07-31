import sys
import os
import re
import glob
from PIL import Image, ImageDraw
import ocrolib
from re import split
import os.path
import json
import numpy as np
import cv2
import imageio
from ..constants import OCRD_TOOL
from shapely.geometry import MultiPoint

import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

import subprocess

from ocrolib import psegutils, morph, sl
from scipy.ndimage.filters import gaussian_filter, uniform_filter, maximum_filter
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_utils import (
    concat_padded, 
    getLogger, 
    MIMETYPE_PAGE, 
    coordinates_for_segment,
    points_from_polygon,
    polygon_from_points
    )

from ocrd_models.ocrd_page import (
    to_xml, 
    AlternativeImageType,
    MetadataItemType,
    LabelsType, LabelType,
    TextRegionType,
    CoordsType,
    TextLineType
    )
    
TOOL = 'ocrd-anybaseocr-textline'
LOG = getLogger('OcrdAnybaseocrTextline')
FALLBACK_IMAGE_GRP = 'OCR-D-IMG-TL'


class OcrdAnybaseocrTextline(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrTextline, self).__init__(*args, **kwargs)

    def addzeros(self, file):
        F = open(file, "r")
        D = F.read()
        D = split("\n", D)
        D = D[:-1]
        F.close()
        F = open(file, "w")
        for d in D:
            d += " 0 0 0 0\n"
            F.write(d)    
    
    def process(self):
        try:
            page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
        oplevel = self.parameter['operation_level']
        
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                    MetadataItemType(type_="processingStep",
                                     name=self.ocrd_tool['steps'][0],
                                     value=TOOL,                                     
                                     Labels=[LabelsType(#externalRef="parameter",
                                                        Label=[LabelType(type_=name,
                                                                         value=self.parameter[name])
                                                               for name in self.parameter.keys()])]))

            page = pcgts.get_Page()
            LOG.info("INPUT FILE %s", input_file.pageId or input_file.ID)
            
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_selector='binarized,deskewed')
            
            if oplevel == 'page':
                LOG.warning("Operation level should be region.")
                self._process_segment(page_image, page,None, page_xywh, page_id, input_file, n)
                
            else:
                regions = page.get_TextRegion()
                if not regions:
                    LOG.warning("Page '%s' contains no text regions", page_id)
                    continue
                for (k, region) in enumerate(regions):
                                       
                    region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh)
        
                    self._process_segment(region_image, page, region, region_xywh, region.id, input_file, k)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(self.input_file_grp, self.output_file_grp)            
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

    def _process_segment(self, page_image, page, textregion, region_xywh, page_id, input_file, n):
        #check for existing text lines and whether to overwrite them
        if textregion.get_TextLine():
            if self.parameter['overwrite']:
                LOG.info('removing existing TextLines in region "%s"', page_id)
                textregion.set_TextLine([])
            else:
                LOG.warning('keeping existing TextLines in region "%s"', page_id)
                return
        
        binary = ocrolib.pil2array(page_image)

        
        if len(binary.shape) > 2:
            binary = np.mean(binary, 2)
        binary = np.array(1-binary/np.amax(binary),'B')
        
        if self.parameter['scale'] == 0:
            scale = psegutils.estimate_scale(binary)
        else:
            scale = self.parameter['scale']
        
        if np.isnan(scale) or scale > 1000.0 or scale < self.parameter['minscale']:
            LOG.warning(str(scale)+": bad scale; skipping!\n" )
            return
        
        segmentation = self.compute_segmentation(binary, scale)
        if np.amax(segmentation) > self.parameter['maxlines']:
            LOG.warning("too many lines %i; skipping!\n", (np.amax(segmentation)))
            return
        lines = psegutils.compute_lines(segmentation, scale)
        order = psegutils.reading_order([l.bounds for l in lines])
        lsort = psegutils.topsort(order)

        # renumber the labels so that they conform to the specs

        nlabels = np.amax(segmentation)+1
        renumber = np.zeros(nlabels, 'i')
        for i, v in enumerate(lsort):
            renumber[lines[v].label] = 0x010000+(i+1)
        segmentation = renumber[segmentation]
        
        lines = [lines[i] for i in lsort]
        cleaned = ocrolib.remove_noise(binary, self.parameter['noise'])
        
        for i, l in enumerate(lines):
            #LOG.info('check this: ') 
            #LOG.info(type(l.bounds))
            #LOG.info(l.bounds)
            #line_points = np.where(l.mask==1)
            #hull = MultiPoint([x for x in zip(line_points[0],line_points[1])]).convex_hull
            #x,y = hull.exterior.coords.xy
            #LOG.info('hull coords x: ',x)
            #LOG.info('hull coords y: ',y)
            
            min_x, max_x = (l.bounds[0].start, l.bounds[0].stop)
            min_y, max_y = (l.bounds[1].start, l.bounds[1].stop)
            
            line_polygon = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            
            #line_polygon = [x for x in zip(y, x)]
            line_polygon = coordinates_for_segment(line_polygon, page_image, region_xywh)
            line_points = points_from_polygon(line_polygon)
            
            img = cleaned[l.bounds[0],l.bounds[1]]
            img = np.array(255*(img>ocrolib.midrange(img)),'B')
            img = 255-img
            img = ocrolib.array2pil(img)
           
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)
        
            file_path = self.workspace.save_image_file(img,
                                   file_id+"_"+str(n)+"_"+str(i),
                                   page_id=page_id,
                                   file_grp=self.image_grp
            )
            ai = AlternativeImageType(filename=file_path, comments=region_xywh['features'])
            line_id = '%s_line%04d' % (page_id, i)
            line = TextLineType(custom='readingOrder {index:'+str(i)+';}', id=line_id, Coords=CoordsType(line_points))
            line.add_AlternativeImage(ai)
            textregion.add_TextLine(line)
            
            #line_test = textregion.get_TextLine()[-1]
            #region_img, region_xy = self.workspace.image_from_segment(line_test, page_image, region_xywh)
            #region_img.save('checkthis.png')
            #cv2.imwrite('checkthis.jpg', region_img)


    def B(self, a):
        if a.dtype == dtype('B'):
            return a
        return np.array(a, 'B')

    ################################################################
    # Column finding.
    ###
    # This attempts to find column separators, either as extended
    # vertical black lines or extended vertical whitespace.
    # It will work fairly well in simple cases, but for unusual
    # documents, you need to tune the parameter.
    ################################################################

    def compute_separators_morph(self, binary, scale):
        """Finds vertical black lines corresponding to column separators."""
        d0 = int(max(5, scale/4))
        d1 = int(max(5, scale))+self.parameter['sepwiden']
        thick = morph.r_dilation(binary, (d0, d1))
        vert = morph.rb_opening(thick, (10*scale, 1))
        vert = morph.r_erosion(vert, (d0//2, self.parameter['sepwiden']))
        vert = morph.select_regions(vert, sl.dim1, min=3, nbest=2*self.parameter['maxseps'])
        vert = morph.select_regions(vert, sl.dim0, min=20*scale, nbest=self.parameter['maxseps'])
        return vert


    def compute_colseps_morph(self, binary, scale, maxseps=3, minheight=20, maxwidth=5):
        """Finds extended vertical whitespace corresponding to column separators
        using morphological operations."""
        boxmap = psegutils.compute_boxmap(binary, scale, (0.4, 5), dtype='B')
        bounds = morph.rb_closing(self.B(boxmap), (int(5*scale), int(5*scale)))
        bounds = maximum(self.B(1-bounds), self.B(boxmap))
        cols = 1-morph.rb_closing(boxmap, (int(20*scale), int(scale)))
        cols = morph.select_regions(cols, sl.aspect, min=self.parameter['csminaspect'])
        cols = morph.select_regions(cols, sl.dim0, min=self.parameter['csminheight']*scale, nbest=self.parameter['maxcolseps'])
        cols = morph.r_erosion(cols, (int(0.5+scale), 0))
        cols = morph.r_dilation(cols, (int(0.5+scale), 0), origin=(int(scale/2)-1, 0))
        return cols


    def compute_colseps_conv(self, binary, scale=1.0):
        """Find column separators by convoluation and
        thresholding."""
        h, w = binary.shape
        
        # find vertical whitespace by thresholding
        smoothed = gaussian_filter(1.0*binary, (scale, scale*0.5))
        smoothed = uniform_filter(smoothed, (5.0*scale, 1))
        thresh = (smoothed < np.amax(smoothed)*0.1)
        
        # find column edges by filtering
        grad = gaussian_filter(1.0*binary, (scale, scale*0.5), order=(0, 1))
        grad = uniform_filter(grad, (10.0*scale, 1))
        grad = (grad > 0.25*np.amax(grad))
        grad1 = morph.select_regions(grad, sl.dim0, min=self.parameter['csminheight']*scale, nbest=self.parameter['maxcolseps']+10)

        x = (1-thresh)*(1-grad1)
        thresh11 = (1-thresh)*x
        
        for r in range(0, len(thresh11)):
            count = 0
            for c in range(0, len(thresh11[0])):
                if(thresh11[r][c] == 1):
                    continue
                count += 1
                if(c != len(thresh11[0])-1 and thresh11[r][c+1] == 1):
                    if(count <= 50):
                        for z in range(c-count, c+1):
                            thresh11[r][z] = 1
                    count = 0

        y = 1-(thresh11*(1-thresh))
        
        # combine edges and whitespace
        seps = np.minimum(thresh, maximum_filter(grad, (int(scale), int(5*scale))))
        seps = maximum_filter(seps, (int(2*scale), 1))
    
        h, w = seps.shape
        smoothed = gaussian_filter(1.0*seps, (scale, scale*0.5))
        smoothed = uniform_filter(smoothed, (5.0*scale, 1))
        seps1 = (smoothed < np.amax(smoothed)*0.1)
        seps1 = 1-seps1
    
        seps1 = (grad)*seps1
    
        for c in range(0, len(seps1[0])):
            count = 0
            for r in range(0, len(seps1)):
                if(seps1[r][c] == 1):
                    continue
                count += 1
                if(r != len(seps1)-1 and seps1[r+1][c] == 1):
                    if(count <= 400):  # by making it 300 u can improve
                        for z in range(r-count, r+1):
                            seps1[z][c] = 1
                    count = 0

    
        seps1 = morph.select_regions(seps1, sl.dim0, min=self.parameter['csminheight']*scale, nbest=self.parameter['maxcolseps']+10)
        seps1 = (seps1*(1-y))+seps1
        for c in range(0, len(seps1[0])):
            for r in range(0, len(seps1)):
                if(seps1[r][c] != 0):
                    seps1[r][c] = 1

        for c in range(0, len(seps1[0])):
            count = 0
            for r in range(0, len(seps1)):
                if(seps1[r][c] == 1):
                    continue
                count += 1
                if(r != len(seps1)-1 and seps1[r+1][c] == 1):
                    if(count <= 350):
                        for z in range(r-count, r+1):
                            seps1[z][c] = 1
                    count = 0

        return seps1


    def compute_colseps(self, binary, scale):
        """Computes column separators either from vertical black lines or whitespace."""
        colseps = self.compute_colseps_conv(binary, scale)
        if self.parameter['blackseps']:
            seps = self.compute_separators_morph(binary, scale)
            colseps = maximum(colseps, seps)
            binary = np.minimum(binary, 1-seps)
        return colseps, binary

    ################################################################
    # Text Line Finding.
    #
    # This identifies the tops and bottoms of text lines by
    # computing gradients and performing some adaptive thresholding.
    # Those components are then used as seeds for the text lines.
    ################################################################

    def compute_gradmaps(self, binary, scale):
        # use gradient filtering to find baselines
        boxmap = psegutils.compute_boxmap(binary, scale, (0.4, 5))
        cleaned = boxmap*binary
        if self.parameter['usegauss']:
            # this uses Gaussians
            grad = gaussian_filter(1.0*cleaned, (self.parameter['vscale']*0.3*scale,
                                                 self.parameter['hscale']*6*scale), order=(1, 0))
        else:
            # this uses non-Gaussian oriented filters
            grad = gaussian_filter(1.0*cleaned, (max(4, self.parameter['vscale']*0.3*scale),
                                                 self.parameter['hscale']*scale), order=(1, 0))
            grad = uniform_filter(grad, (self.parameter['vscale'], self.parameter['hscale']*6*scale))
        bottom = ocrolib.norm_max((grad < 0)*(-grad))
        top = ocrolib.norm_max((grad > 0)*grad)
        testseeds = np.zeros(binary.shape, 'i')
        return bottom, top, boxmap


    def compute_line_seeds(self, binary, bottom, top, colseps, scale):
        """Base on gradient maps, computes candidates for baselines
        and xheights.  Then, it marks the regions between the two
        as a line seed."""
        t = self.parameter['threshold'] 
        vrange = int(self.parameter['vscale']*scale)
        bmarked = maximum_filter(bottom == maximum_filter(bottom, (vrange, 0)), (2, 2))
        bmarked *= np.array((bottom > t*np.amax(bottom)*t)*(1-colseps), dtype=bool)
        tmarked = maximum_filter(top == maximum_filter(top, (vrange, 0)), (2, 2))
        tmarked *= np.array((top > t*np.amax(top)*t/2)*(1-colseps), dtype=bool)
        tmarked = maximum_filter(tmarked, (1, 20))
        testseeds = np.zeros(binary.shape, 'i')
        seeds = np.zeros(binary.shape, 'i')
        delta = max(3, int(scale/2))
        for x in range(bmarked.shape[1]):
            transitions = sorted([(y, 1) for y in psegutils.find(bmarked[:, x])]+[(y, 0) for y in psegutils.find(tmarked[:, x])])[::-1]
            transitions += [(0, 0)]
            for l in range(len(transitions)-1):
                y0, s0 = transitions[l]
                if s0 == 0:
                    continue
                seeds[y0-delta:y0, x] = 1
                y1, s1 = transitions[l+1]
                if s1 == 0 and (y0-y1) < 5*scale:
                    seeds[y1:y0, x] = 1
        seeds = maximum_filter(seeds, (1, int(1+scale)))
        seeds *= (1-colseps)
        seeds, _ = morph.label(seeds)
        return seeds

    
    ################################################################
    # The complete line segmentation process.
    ################################################################

    def remove_hlines(self, binary, scale, maxsize=10):
        labels, _ = morph.label(binary)
        objects = morph.find_objects(labels)
        for i, b in enumerate(objects):
            if sl.width(b) > maxsize*scale:
                labels[b][labels[b] == i+1] = 0
        return np.array(labels != 0, 'B')


    def compute_segmentation(self, binary, scale):
        """Given a binary image, compute a complete segmentation into
        lines, computing both columns and text lines."""
        binary = np.array(binary, 'B')
        # start by removing horizontal black lines, which only
        # interfere with the rest of the page segmentation
        binary = self.remove_hlines(binary, scale)
        
        # do the column finding
        colseps, binary = self.compute_colseps(binary, scale)
        
        # now compute the text line seeds
        bottom, top, boxmap = self.compute_gradmaps(binary, scale)
        seeds = self.compute_line_seeds(binary, bottom, top, colseps, scale)
        # spread the text line seeds to all the remaining
        # components
        llabels = morph.propagate_labels(boxmap, seeds, conflict=0)
        spread = morph.spread_labels(seeds, maxdist=scale)
        llabels = np.where(llabels > 0, llabels, spread*binary)
        segmentation = llabels*binary
        return segmentation

        
    
@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrTextline, *args, **kwargs)
