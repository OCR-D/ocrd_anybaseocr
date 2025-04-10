from functools import cached_property
import os
import sys
import pickle
from typing import Dict, List, Optional, Union
import numpy as np 
import warnings
from ocrd.mets_server import ClientSideOcrdMets

from ocrd_models import OcrdFileType
warnings.filterwarnings('ignore',category=FutureWarning) 
from collections import defaultdict

import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ocrd import OcrdPage, OcrdPageResult, Processor, Workspace
from ocrd_modelfactory import page_from_file
from ocrd_utils import getLogger, resource_filename

from pathlib import Path
from PIL import Image

from lxml import etree as ET

from ocrd_models.constants import (
    NAMESPACES as NS,
    TAG_METS_DIV,
    TAG_METS_STRUCTMAP,
)

from ..tensorflow_importer import tf
from ..utils import pil2array
from tensorflow.keras.models import load_model

TAG_METS_STRUCTLINK = '{%s}structLink' % NS['mets']
TAG_METS_SMLINK = '{%s}smLink' % NS['mets']

class OcrdAnybaseocrLayoutAnalyser(Processor):

    max_workers = 1 # Tensorflow context cannot be shared across forked workers

    @cached_property
    def executable(self):
        return 'ocrd-anybaseocr-layout-analysis'

    @property
    def moduledir(self):
        return resource_filename(self.module, 'models')

    def setup(self):
        devices = tf.config.list_physical_devices('GPU')
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        if not devices:
            self.logger.warning("Your system has no CUDA installed. No GPU detected.")

        assert self.parameter
        model_path = Path(self.resolve_resource(self.parameter['model_path']))
        class_mapper_path = Path(self.resolve_resource(self.parameter['class_mapping_path']))
        self.logger.info('Loading model from file %s', str(model_path))
        self.model = self.create_model(str(model_path))
        # load the mapping
        with open(str(class_mapper_path), "rb") as pickle_in:
            class_indices = pickle.load(pickle_in)
        self.label_mapping: Dict[int, str] = dict((v,k) for k,v in class_indices.items())
        self.reset()

    def reset(self):
        self.page_labels: Dict[str, List[str]] = {} # Mapping of page_id to  detected labels

    def process_workspace(self, workspace: Workspace) -> None:
        super().process_workspace(workspace)
        writeable_workspace = workspace
        if isinstance(workspace.mets, ClientSideOcrdMets):
            # (changes could have accumulated in prior processing step)
            workspace.save_mets()
            # instantiate (read and parse) METS from disk (read-only, metadata are constant)
            writeable_workspace = Workspace(workspace.resolver, workspace.directory,
                           mets_basename=os.path.basename(workspace.mets_target))
        self.create_logmap_smlink(writeable_workspace)
        self.add_log_divs()
        writeable_workspace.save_mets()
        if isinstance(workspace.mets, ClientSideOcrdMets):
            workspace.mets.reload()
        self.reset()

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
        assert len(input_files) == 1 and input_files[0]
        input_file = input_files[0]
        page_id = input_file.pageId
        self._base_logger.info("processing page %s", page_id)
        self._base_logger.info(f"parsing file {input_file.ID} for page {page_id}")
        pcgts = page_from_file(input_file)
        page = pcgts.get_Page()
        page_image, _, _ = self.workspace.image_from_page(page, page_id, feature_selector='binarized')
        img_array = pil2array(page_image.resize((500, 600), Image.Resampling.LANCZOS))
        img_array = img_array / 255
        img_array = img_array[np.newaxis, :, :, np.newaxis]
        self.page_labels[page_id] = self._predict(img_array)

    def shutdown(self) -> None:
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'label_mapping'):
            del self.label_mapping
        return super().shutdown()

    def create_model(self, path):
        #model_name='inception_v3', def_weights=True, num_classes=34, input_size=(600, 500, 1)):
        '''load Tensorflow model from path'''
        return load_model(path)

    def _predict(self, img_array) -> List[str]:
        # shape should be 1,600,500 for keras
        pred = self.model.predict(img_array)
        pred = np.array(pred)
        # multi-label predictions
        if len(pred.shape) > 2:
            pred = np.squeeze(pred)
            pred = pred.T
        preds = (pred>=0.5)
        predictions = []
        for index, cls in enumerate(preds):
            #self.logger.debug("%d[%s]: %f", index, self.label_mapping[index], cls)
            if cls:
                predictions.append(self.label_mapping[index])
        if len(predictions) == 0:
            # if no prediction get the maximum one
            predictions.append(self.label_mapping[np.argmax(pred)])
            #predictions.append('page') # default label
        self.logger.info(predictions)
        return predictions

    def img_resize(self, image_path):
        size = 600, 500
        img = Image.open(image_path)
        return img.thumbnail(size, Image.Resampling.LANCZOS)

    def add_log_divs(self):
        # counter for new logical mets:div key
        logID = 0
        # dict from label into counter:
        # keep track of previous keys for labels other then chapter or section
        logIDs = defaultdict(int)
        # dict from label into div elements:
        # keep track of previous keys (and init from existing divs)
        log_divs = [div
                    for div in self.log_map.iterdescendants(TAG_METS_DIV)
                    # get existing non-structural divs (i.e. top-level like volume/issue/monograph):
                    if (div_type := div.get('TYPE')) not in self.label_mapping.values()]
        first = log_divs[-1].get('TYPE').lower() if log_divs else None
        log_divs = {div.get('TYPE').lower(): div
                    for div in log_divs}
        prev_labels = []
        for page_id, labels in self.page_labels.items():
            for label in labels:
                # key to use for current page (create new mets:div if negative)
                page_logID = -1
                # check if label is page skip
                if label != "page":
                    # if not page, chapter and section then it's something old
                    if label not in ["chapter", "section"]:
                        if label in prev_labels:
                            # continue with div from previous page
                            page_logID = logIDs[label]

                        if label == 'binding':
                            parent_node = self.log_map
                        elif label in ['cover', 'endsheet', 'paste_down']:
                            # get the link for master node
                            parent_node = log_divs['binding']
                        elif label != 'title_page' and first is not None:
                            parent_node = log_divs[first]
                        else:
                            parent_node = self.log_map
                    else:
                        if first is None:
                            first = label
                            parent_node = self.log_map
                        elif first == label:
                            parent_node = self.log_map
                        else:
                            parent_node = log_divs[first]
                else:
                    # label is (normal / follow-up content) page
                    if logIDs['chapter'] > logIDs['section']:
                        page_logID = logIDs['chapter']
                    if logIDs['section'] > logIDs['chapter']:
                        page_logID = logIDs['section']
                    if logIDs['chapter']==0 and logIDs['section']==0:

                        # if both chapter and section dont exist
                        if first is None:
                            first = 'chapter'
                            parent_node = self.log_map
                        # rs: not sure about the remaining branches (cf. 0bbcb66b in #73)
                        elif first == label:
                            parent_node = self.log_map
                        else:
                            parent_node = log_divs[first]

                if page_logID < 0:
                    log_div = ET.SubElement(parent_node, TAG_METS_DIV)
                    log_div.set('TYPE', str(label))
                    log_div.set('ID', "LOG_%04d" % logID)
                    log_divs[label] = log_div # store the link
                    #if label!='chapter' and label!='section':
                    logIDs[label] = logID
                    page_logID = logID
                    logID += 1
                    self.logger.info("added %s[%s] to %s[%s]", "LOG_%04d" % page_logID, label,
                                     parent_node.get('ID'), parent_node.get('TYPE'))

                smLink = ET.SubElement(self.link, TAG_METS_SMLINK)
                smLink.set('{'+NS['xlink']+'}'+'from', "LOG_%04d" % page_logID)
                smLink.set('{'+NS['xlink']+'}'+'to', page_id)
                self.logger.debug("smlinked %s → %s", "LOG_%04d" % page_logID, page_id)

            prev_labels = labels

    def create_logmap_smlink(self, workspace):
        # NOTE: workspace is not necessarily self.workspace here due to METS Server
        el_root = workspace.mets._tree.getroot()
        if (log_map := el_root.find(TAG_METS_STRUCTMAP + '[@TYPE="LOGICAL"]')) is None:
            log_map = ET.SubElement(el_root, TAG_METS_STRUCTMAP)
            log_map.set('TYPE', 'LOGICAL')
        else:
            self.logger.info('LOGICAL structMap already exists, adding to it')
        if (link := el_root.find(TAG_METS_STRUCTLINK)) is None:
            link = ET.SubElement(el_root, TAG_METS_STRUCTLINK)
        self.link = link
        self.log_map = log_map

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrLayoutAnalyser, *args, **kwargs)
