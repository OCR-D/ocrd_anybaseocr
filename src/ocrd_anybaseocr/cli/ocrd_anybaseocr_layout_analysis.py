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

    def setup(self):
        if not tf.config.list_physical_devices('GPU'):
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
        self.last_result = [] 
        self.logID = 0 # counter for new key
        self.logIDs = defaultdict(int) # dict to keep track of previous keys for labels other then chapter or section
        self.log_id = 0 # var to keep the current ongoing key
        self.log_links = {}
        self.first = None
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
        for page_id, labels in self.page_labels.items():
            self.write_to_mets(labels, page_id)
        writeable_workspace.save_mets()
        if isinstance(workspace.mets, ClientSideOcrdMets):
            workspace.mets.reload()
        self.reset()

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
        self.logger.info("Overridden process_page_file")
        assert len(input_files) == 1 and input_files[0]
        input_file = input_files[0]
        page_id = input_file.pageId
        pcgts = page_from_file(input_file)
        page = pcgts.get_Page()
        self.logger.info("INPUT FILE %s", page_id)
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
        self.logger.debug(predictions)
        return predictions

    def img_resize(self, image_path):
        size = 600, 500
        img = Image.open(image_path)
        return img.thumbnail(size, Image.Resampling.LANCZOS)

    def write_to_mets(self, labels: List[str], pageID: str):
        for label in labels:
            create_new_logical = False
            # check if label is page skip 
            if label !="page":
                # if not page, chapter and section then its something old
                if label!="chapter" and label!="section":
                    if label in self.last_result:
                        self.log_id = self.logIDs[label]
                    else:
                        create_new_logical = True

                    if label =='binding':
                        parent_node = self.log_map

                    if label=='cover' or label=='endsheet' or label=='paste_down':
                        # get the link for master node
                        parent_node = self.log_links['binding']
                    else:
                        if self.first is not None and label!='title_page':
                            parent_node = self.log_links[self.first]
                        else:
                            parent_node = self.log_map

                else:
                    create_new_logical = True

                    if self.first is None:
                        self.first = label
                        parent_node = self.log_map
                    else:
                        if self.first == label:
                            parent_node = self.log_map
                        else:
                            parent_node = self.log_links[self.first]

            else:
                if self.logIDs['chapter'] > self.logIDs['section']:
                    self.log_id = self.logIDs['chapter']

                if self.logIDs['section'] > self.logIDs['chapter']:
                    self.log_id = self.logIDs['section']

                if self.logIDs['chapter']==0 and self.logIDs['section']==0:

                    create_new_logical = True

                    # if both chapter and section dont exist
                    if self.first is None:
                        self.first = 'chapter'
                        parent_node = self.log_map
                    # rs: not sure about the remaining branches (cf. #73)
                    elif self.first == label:
                        parent_node = self.log_map
                    else:
                        parent_node = self.log_links[self.first]

            if create_new_logical:
                log_div = ET.SubElement(parent_node, TAG_METS_DIV)
                log_div.set('TYPE', str(label))
                log_div.set('ID', "LOG_%04d" % self.logID)
                self.log_links[label] = log_div # store the link 
                #if label!='chapter' and label!='section':
                self.logIDs[label] = self.logID
                self.log_id = self.logID
                self.logID += 1
                self.logger.debug("added %s to %s", "LOG_%04d" % self.log_id, parent_node)

            smLink = ET.SubElement(self.link, TAG_METS_SMLINK)
            smLink.set('{'+NS['xlink']+'}'+'to', pageID)
            smLink.set('{'+NS['xlink']+'}'+'from', "LOG_%04d" % self.log_id)
            self.logger.debug("smlinked %s → %s", "LOG_%04d" % self.log_id, pageID)

        self.last_result = labels

    def create_logmap_smlink(self, workspace):
        # NOTE: workspace is not necessarily self.workspace here due to METS Server
        el_root = workspace.mets._tree.getroot()
        log_map = el_root.find('mets:structMap[@TYPE="LOGICAL"]', NS)
        if log_map is None:
            log_map = ET.SubElement(el_root, TAG_METS_STRUCTMAP)
            log_map.set('TYPE', 'LOGICAL')
        else:
            self.logger.info('LOGICAL structMap already exists, adding to it')
        link = el_root.find('mets:structLink', NS)
        if link is None:
            link = ET.SubElement(el_root, TAG_METS_STRUCTLINK)
        self.link = link
        self.log_map = log_map

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrLayoutAnalyser, *args, **kwargs)
