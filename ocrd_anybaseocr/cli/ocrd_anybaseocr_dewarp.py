import torch
import sys
import os
import shutil

from ..constants import OCRD_TOOL

from ocrd import Processor

from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE
from ocrd_modelfactory import page_from_file
from pylab import array
from pathlib import Path
from PIL import Image
import ocrolib
from ocrd_models.ocrd_page import (
    to_xml, 
    AlternativeImageType,
    MetadataItemType,
    LabelsType, LabelType
    )

TOOL = 'ocrd-anybaseocr-dewarp'
LOG = getLogger('OcrdAnybaseocrDewarper')
FALLBACK_IMAGE_GRP = 'OCR-D-IMG-DEWARP'

class OcrdAnybaseocrDewarper(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']        
        super(OcrdAnybaseocrDewarper, self).__init__(*args, **kwargs)

    def prepare_options(self, path):
        sys.path.append(path)
        from options.test_options import TestOptions
        from models.models import create_model
        sys.argv = [sys.argv[0]]
        os.mkdir(self.input_file_grp+"/test_A/")
        opt = TestOptions().parse(save=False)
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.rood_dir = self.input_file_grp # make into proper path
        opt.checkpoints_dir = self.parameter['checkpoint_dir'] 
        opt.dataroot = self.input_file_grp
        opt.name = self.parameter['model_name']
        opt.label_nc = 0
        opt.no_instance = True
        opt.resize_or_crop = self.parameter['imgresize']
        opt.n_blocks_global = 10
        opt.n_local_enhancers = 2
        opt.gpu_ids = [self.parameter['gpu_id']]
        opt.loadSize = self.parameter['resizeHeight']
        opt.fineSize = self.parameter['resizeWidth']
        
        model = create_model(opt)
    
        return opt, model

    def prepare_data(self, opt, page_img, path):

        sys.path.append(path)
        from data.data_loader import CreateDataLoader
        
        data_loader = CreateDataLoader(opt)
        data_loader.dataset.A_paths = [page_img.filename]
        data_loader.dataset.dataset_size = len(data_loader.dataset.A_paths)
        data_loader.dataloader = torch.utils.data.DataLoader(data_loader.dataset,
                                                             batch_size=opt.batchSize,
                                                             shuffle=not opt.serial_batches,
                                                             num_workers=int(opt.nThreads))
        dataset = data_loader.load_data()
        return dataset   
        # test
        
    def process(self):
        try:
            page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
        if not torch.cuda.is_available():
            LOG.warning("torch cannot detect CUDA installation.")
            self.parameter['gpu_id'] = 'cpu'

        path = self.parameter['pix2pixHD']
        if not Path(path).is_dir():
            LOG.error("""\
                NVIDIA's pix2pixHD was not found at '%s'. Make sure the `pix2pixHD` parameter 
                in ocrd-tools.json points to the local path to the cloned pix2pixHD repository.

                pix2pixHD can be downloaded from https://github.com/NVIDIA/pix2pixHD
                """ % path)
            sys.exit(1)
        model_file_path = os.path.join(path, 'checkpoints/latest_net_G.pth')    
        if not Path(model_file_path).is_file():
            LOG.error("""\
                pix2pixHD model file was not found at '%s'. Make sure the this file exists.
                """ % model_file_path)
            sys.exit(1)
                
        opt, model = self.prepare_options(path)
        
        
        oplevel = self.parameter['operation_level']
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %s", page_id)
            
            pcgts = page_from_file(self.workspace.download_file(input_file))
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                    MetadataItemType(type_="processingStep",
                                     name=self.ocrd_tool['steps'][0],
                                     value=TOOL,                                     
                                     Labels=[LabelsType(#externalRef="parameters",
                                                        Label=[LabelType(type_=name,
                                                                         value=self.parameter[name])
                                                               for name in self.parameter.keys()])]))

            page = pcgts.get_Page()
            
            page_image, page_xywh, page_image_info = self.workspace.image_from_page(page, page_id, feature_filter='dewarped', feature_selector='binarized') # images should be deskewed and cropped
            if oplevel == 'page':
                dataset = self.prepare_data(opt, page_image, path)
                orig_img_size = page_image.size
                self._process_segment(model, dataset, page, page_xywh, page_id, input_file, orig_img_size, n)
            else:
                regions = page.get_TextRegion() + page.get_TableRegion() #get all regions?
                if not regions: 
                    LOG.warning("Page '%s' contains no text regions", page_id)
                for (k, region) in enumerate(regions):
                    region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh)            
                    # TODO: not tested on regions
                    # TODO: region has to exist as a physical file to be processed by pix2pixHD
                    dataset = self.prepare_data(opt, region_image, path)
                    orig_img_size = region_image.size
                    self._process_segment(model, dataset, page, region_xywh, region.id, input_file, orig_img_size, n)
           
            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(self.input_file_grp, page_grp)            
            if file_id == input_file.ID:
                file_id = concat_padded(page_grp, n)                
            self.workspace.add_file(
                ID=file_id,
                file_grp=page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(page_grp,
                                        file_id + '.xml'),
                content=to_xml(pcgts).encode('utf-8'),
                force=self.parameter['force']
            )
        os.rmdir(self.input_file_grp+"/test_A/") #FIXME: better way of deleting a temp_dir?
        

    def _process_segment(self, model, dataset, page, page_xywh, page_id, input_file, orig_img_size, n):
        for i, data in enumerate(dataset):
            w,h = orig_img_size
            generated = model.inference(data['label'], data['inst'], data['image'])
            dewarped = array(generated.data[0].permute(1,2,0).detach().cpu())
            bin_array = array(255*(dewarped>ocrolib.midrange(dewarped)),'B')
            dewarped = ocrolib.array2pil(bin_array)
            dewarped = dewarped.resize((w,h))                        
            
            page_xywh['features'] += ',dewarped'  
            
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.image_grp, n)
        
            file_path = self.workspace.save_image_file(dewarped,
                                   file_id,
                                   page_id=page_id,
                                   file_grp=self.image_grp,
                                   force=self.parameter['force']
                )     
            page.add_AlternativeImage(AlternativeImageType(filename=file_path, comments=page_xywh['features']))
        

