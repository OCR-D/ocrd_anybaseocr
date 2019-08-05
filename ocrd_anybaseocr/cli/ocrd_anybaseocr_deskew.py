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


import numpy as np
from pylab import amin, amax, linspace, mean, var, plot, ginput, ones, clip, imshow
from scipy.ndimage import filters, interpolation, morphology
from scipy import stats
import ocrolib
from ..utils import print_info
from ..constants import OCRD_TOOL

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml,  TextRegionType
from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE
from ocrd_models.ocrd_page_generateds import RegionType


class OcrdAnybaseocrDeskewer(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-deskew']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrDeskewer, self).__init__(*args, **kwargs)

    def estimate_skew_angle(self, image, angles):
        param = self.parameter
        estimates = []

        for a in angles:
            v = mean(interpolation.rotate(
                image, a, order=0, mode='constant'), axis=1)
            v = var(v)
            estimates.append((v, a))
        if param['debug'] > 0:
            plot([y for x, y in estimates], [x for x, y in estimates])
            ginput(1, param['debug'])
        _, a = max(estimates)
        return a

    def process(self):
        for (n, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            fname = pcgts.get_Page().imageFilename
            img = self.workspace.resolve_image_as_pil(fname)
            param = self.parameter
            base, _ = ocrolib.allsplitext(fname)
            #basefile = ocrolib.allsplitext(os.path.basename(fpath))[0]

            if param['parallel'] < 2:
                print_info("=== %s " % (fname))
            raw = ocrolib.read_image_gray(img.filename)

            flat = raw
            #flat = np.array(binImg)
            # estimate skew angle and rotate
            if param['maxskew'] > 0:
                if param['parallel'] < 2:
                    print_info("estimating skew angle")
                d0, d1 = flat.shape
                o0, o1 = int(param['bignore']*d0), int(param['bignore']*d1)
                flat = amax(flat)-flat
                flat -= amin(flat)
                est = flat[o0:d0-o0, o1:d1-o1]
                ma = param['maxskew']
                ms = int(2*param['maxskew']*param['skewsteps'])
                angle = self.estimate_skew_angle(est, linspace(-ma, ma, ms+1))
                flat = interpolation.rotate(
                    flat, angle, mode='constant', reshape=0)
                flat = amax(flat)-flat
            else:
                angle = 0

            # self.write_angles_to_pageXML(base,angle)
            # estimate low and high thresholds
            if param['parallel'] < 2:
                print_info("estimating thresholds")
            d0, d1 = flat.shape
            o0, o1 = int(param['bignore']*d0), int(param['bignore']*d1)
            est = flat[o0:d0-o0, o1:d1-o1]
            if param['escale'] > 0:
                # by default, we use only regions that contain
                # significant variance; this makes the percentile
                # based low and high estimates more reliable
                e = param['escale']
                v = est-filters.gaussian_filter(est, e*20.0)
                v = filters.gaussian_filter(v**2, e*20.0)**0.5
                v = (v > 0.3*amax(v))
                v = morphology.binary_dilation(
                    v, structure=ones((int(e*50), 1)))
                v = morphology.binary_dilation(
                    v, structure=ones((1, int(e*50))))
                if param['debug'] > 0:
                    imshow(v)
                    ginput(1, param['debug'])
                est = est[v]
            lo = stats.scoreatpercentile(est.ravel(), param['lo'])
            hi = stats.scoreatpercentile(est.ravel(), param['hi'])
            # rescale the image to get the gray scale image
            if param['parallel'] < 2:
                print_info("rescaling")
            flat -= lo
            flat /= (hi-lo)
            flat = clip(flat, 0, 1)
            if param['debug'] > 0:
                imshow(flat, vmin=0, vmax=1)
                ginput(1, param['debug'])
            deskewed = 1*(flat > param['threshold'])

            # output the normalized grayscale and the thresholded images
            print_info("%s lo-hi (%.2f %.2f) angle %4.1f" %
                       (pcgts.get_Page().imageFilename, lo, hi, angle))
            if param['parallel'] < 2:
                print_info("writing")
            ocrolib.write_image_binary(base+".ds.png", deskewed)

            orientation = -angle
            orientation = 180 - (180 - orientation) % 360
            pcgts.get_Page().set_orientation(orientation)

            ID = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=ID,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype="image/png",
                url=base + ".ds.png",
                local_filename='%s/%s' % (self.output_file_grp, ID),
                content=to_xml(pcgts).encode('utf-8')
            )
