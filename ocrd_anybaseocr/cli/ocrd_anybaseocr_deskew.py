# pylint: disable=invalid-name
# ======================================================================
# ====================================
# README file for Skew Correction component
# ====================================

# Filename : ocrd-anyBaseOCR-deskew.py

# Author: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Responsible: Syed Saqib Bukhari, Mohammad Mohsin Reza, Md. Ajraf Rakib
# Contact Email: Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de, Md_ajraf.rakib@dfki.de
# Note:
# 1) this work has been done in DFKI, Kaiserslautern, Germany.
# 2) The parameters values are read from ocrd-anyBaseOCR-parameter.json file. The values can be changed in that file.
# 3) The command line IO usage is based on "OCR-D" project guidelines (https://ocr-d.github.io/). A sample image file (samples/becker_quaestio_1586_00013.tif) and mets.xml (mets.xml) are provided. The sequence of operations is: binarization, deskewing, cropping and dewarping (or can also be: binarization, dewarping, deskewing, and cropping; depends upon use-case).

# *********** Method Behaviour ********************
# This function takes a document image as input and do the skew correction of that document.
# *********** Method Behaviour ********************

# *********** LICENSE ********************
# License: ocropus-nlbin.py (from https://github.com/tmbdev/ocropy/) contains both functionalities: binarization and skew correction.
# This method (ocrd-anyBaseOCR-deskew.py) only contains the skew correction functionality of ocropus-nlbin.py.
# It still has the same licenses as ocropus-nlbin, i.e Apache 2.0 (the ocropy license details are pasted below).
# This file is dependend on ocrolib library which comes from https://github.com/tmbdev/ocropy/.

# Copyright 2014 Thomas M. Breuel

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
# limitations under the License.

# *********** LICENSE ********************
# ======================================================================
#!/usr/bin/env python

import os
import numpy as np
from pylab import amin,array, amax, linspace, mean, var, plot, ginput, ones, clip, imshow
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import ocrolib
from ..constants import OCRD_TOOL
import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    to_xml,
    AlternativeImageType,
    )
from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE,
    make_file_id,
    assert_file_grp_cardinality
)

TOOL = 'ocrd-anybaseocr-deskew'

class OcrdAnybaseocrDeskewer(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrDeskewer, self).__init__(*args, **kwargs)

    def estimate_skew_angle(self, image, angles):
        
        estimates = []
        
        for a in angles:
            v = mean(interpolation.rotate(
                image, a, order=0, mode='constant'), axis=1)
            v = var(v)
            estimates.append((v, a))
        if self.parameter['debug'] > 0:
            plot([y for x, y in estimates], [x for x, y in estimates])
            ginput(1, self.parameter['debug'])
        _, a = max(estimates)
        return a

    def process(self):
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        oplevel = self.parameter['operation_level']

        LOG = getLogger('OcrdAnybaseocrDeskewer')

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID 
            
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            angle = page.get_orientation()
            if angle:
                LOG.warning('Overwriting existing deskewing angle: %i', angle)
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_filter='deskewed',feature_selector='binarized')
            
                        
            if oplevel=="page":
                self._process_segment(page_image, page, page_xywh, page_id, input_file, n) 
            else:
                LOG.warning('Operation level %s, but should be "page".', oplevel)
                break
            
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
    
    def _process_segment(self,page_image, page, page_xywh, page_id, input_file, n):                
        LOG = getLogger('OcrdAnybaseocrDeskewer')
        raw = ocrolib.pil2array(page_image)
        flat = raw.astype("float64")

        # estimate skew angle and rotate
        if self.parameter['maxskew'] > 0:
            if self.parameter['parallel'] < 2:
                LOG.info("Estimating Skew Angle")
            d0, d1 = flat.shape
            o0, o1 = int(self.parameter['bignore']*d0), int(self.parameter['bignore']*d1)
            flat = amax(flat)-flat
            flat -= amin(flat)
            est = flat[o0:d0-o0, o1:d1-o1]
            ma = self.parameter['maxskew']
            ms = int(2*self.parameter['maxskew']*self.parameter['skewsteps'])
            angle = self.estimate_skew_angle(est, linspace(-ma, ma, ms+1))
            flat = interpolation.rotate(
                flat, angle, mode='constant', reshape=0)
            flat = amax(flat)-flat
        else:
            angle = 0

        # self.write_angles_to_pageXML(base,angle)
        # estimate low and high thresholds
        if self.parameter['parallel'] < 2:
            LOG.info("Estimating Thresholds")
        d0, d1 = flat.shape
        o0, o1 = int(self.parameter['bignore']*d0), int(self.parameter['bignore']*d1)
        est = flat[o0:d0-o0, o1:d1-o1]
        if self.parameter['escale'] > 0:
            # by default, we use only regions that contain
            # significant variance; this makes the percentile
            # based low and high estimates more reliable
            e = self.parameter['escale']
            v = est-filters.gaussian_filter(est, e*20.0)
            v = filters.gaussian_filter(v**2, e*20.0)**0.5
            v = (v > 0.3*amax(v))
            v = morphology.binary_dilation(
                v, structure=ones((int(e*50), 1)))
            v = morphology.binary_dilation(
                v, structure=ones((1, int(e*50))))
            if self.parameter['debug'] > 0:
                imshow(v)
                ginput(1, self.parameter['debug'])
            est = est[v]
        lo = stats.scoreatpercentile(est.ravel(), self.parameter['lo'])
        hi = stats.scoreatpercentile(est.ravel(), self.parameter['hi'])

        # rescale the image to get the gray scale image
        if self.parameter['parallel'] < 2:
            LOG.info("Rescaling")
        flat -= lo
        flat /= (hi-lo)
        flat = clip(flat, 0, 1)
        if self.parameter['debug'] > 0:
            imshow(flat, vmin=0, vmax=1)
            ginput(1, self.parameter['debug'])
        deskewed = 1*(flat > self.parameter['threshold'])

        # output the normalized grayscale and the thresholded images
        #LOG.info("%s lo-hi (%.2f %.2f) angle %4.1f" %(lo, hi, angle))

        #TODO: Need some clarification as the results effect the following pre-processing steps.
        #orientation = -angle
        #orientation = 180 - ((180 - orientation) % 360)
        
        if angle is None: # FIXME: quick fix to prevent angle of "none"
            angle = 0
        
        page.set_orientation(angle)
        
        page_xywh['features'] += ',deskewed'
        bin_array = array(255*(deskewed>ocrolib.midrange(deskewed)),'B')
        page_image = ocrolib.array2pil(bin_array)
        
        file_id = make_file_id(input_file, self.output_file_grp)
        file_path = self.workspace.save_image_file(page_image,
                               file_id + '-IMG',
                               page_id=page_id,
                               file_grp=self.output_file_grp
        )        
        page.add_AlternativeImage(AlternativeImageType(filename=file_path, comments=page_xywh['features']))
        
        
        
        

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDeskewer, *args, **kwargs)    
