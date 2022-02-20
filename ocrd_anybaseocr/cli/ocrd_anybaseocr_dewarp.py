# pylint: disable=wrong-import-order, import-error, too-few-public-methods
# pylint: disable=too-many-locals, line-too-long, invalid-name, too-many-arguments
# pylint: disable=missing-docstring

import sys
import os
from pathlib import Path
from PIL import Image
import click
import torch
import numpy as np

import ocrolib
from ocrd import Processor
from ocrd_models.ocrd_page import to_xml, AlternativeImageType
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_utils import (
    getLogger,
    MIMETYPE_PAGE,
    assert_file_grp_cardinality,
    make_file_id
)
from ocrd_modelfactory import page_from_file
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ..constants import OCRD_TOOL
from ..pix2pixhd.options.test_options import TestOptions
from ..pix2pixhd.models.models import create_model
from ..pix2pixhd.data.base_dataset import BaseDataset, get_params, get_transform
from ..pix2pixhd.util.util import tensor2im

TOOL = 'ocrd-anybaseocr-dewarp'

class TestDataset(BaseDataset):
    # adopted from pix2pixhd.data.AlignDataset for our TestOptions
    # but with in-memory Image
    def __init__(self, opt, images):
        super().__init__()
        self.opt = opt
        self.images = images
    def __getitem__(self, index):
        image = self.images[index]
        param = get_params(self.opt, image.size)
        trans = get_transform(self.opt, param)
        tensor = trans(image.convert('RGB'))
        return {'label': tensor, 'path': '',
                'inst': 0, 'image': 0, 'feat': 0}
    def __len__(self):
        return len(self.images) // self.opt.batchSize * self.opt.batchSize

def prepare_data(opt, page_img):
    # todo: make asynchronous (all pages for continuous quasi-parallel decoding)
    dataset = TestDataset(opt, [page_img])
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=opt.batchSize,
                                       shuffle=not opt.serial_batches,
                                       num_workers=int(opt.nThreads))

def prepare_options(gpu_id, model_path, resize_or_crop, loadSize, fineSize):
    LOG = getLogger('OcrdAnybaseocrDewarper')
    # XXX https://github.com/OCR-D/ocrd_anybaseocr/pull/62#discussion_r450232164
    # The problem was with how BaseOptions.parse is implemented in pix2pixHD based on
    # argparse. I cannot explain why but the approach to let pix2pixHD fill the
    # TestOptions instance with argparse default values and then modifying the
    # instance did not work, the overrides were simply ignored. The only way I got
    # pix2pixHD to reliably pick up the overrides was this sys.argv approach. It's
    # ugly, true, but so is using argparse as an API. At least this way, it is
    # uniform as you say.
    sys.argv = ['python']
    sys.argv.extend(['--gpu_ids', str(gpu_id)])
    sys.argv.extend(['--nThreads', str(1)])   # test code only supports nThreads = 1
    sys.argv.extend(['--batchSize', str(1)])  # test code only supports batchSize = 1
    sys.argv.extend(['--serial_batches'])  # no shuffle
    sys.argv.extend(['--no_flip'])  # no flip
    sys.argv.extend(['--dataroot', dataroot])
    sys.argv.extend(['--checkpoints_dir', str(model_path.parents[1])])
    sys.argv.extend(['--name', model_path.parents[0].name])
    sys.argv.extend(['--label_nc', str(0)])
    sys.argv.extend(['--no_instance'])
    sys.argv.extend(['--resize_or_crop', resize_or_crop])
    sys.argv.extend(['--n_blocks_global', str(10)])
    sys.argv.extend(['--n_local_enhancers', str(2)])
    sys.argv.extend(['--loadSize', str(loadSize)])
    sys.argv.extend(['--fineSize', str(fineSize)])
    sys.argv.extend(['--model', 'pix2pixHD'])
    sys.argv.extend(['--verbose'])
    LOG.debug("Options passed to pix2pixHD: %s", sys.argv)
    opt = TestOptions()
    opt.initialize()
    opt = opt.parse(save=False)

    model = create_model(opt)

    return opt, model

class OcrdAnybaseocrDewarper(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrDewarper, self).__init__(*args, **kwargs)


    def process(self):
        LOG = getLogger('OcrdAnybaseocrDewarper')

        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        if self.parameter['gpu_id'] > -1 and not torch.cuda.is_available():
            LOG.warning("torch cannot detect CUDA installation.")
            self.parameter['gpu_id'] = -1

        model_path = Path(self.resolve_resource(self.parameter['model_path']))
        if not model_path.is_file():
            LOG.error("""\
                    pix2pixHD model file was not found at '%s'. Make sure this file exists.
                """ % model_path)
            sys.exit(1)

        opt, model = prepare_options(
            gpu_id=self.parameter['gpu_id'],
            dataroot=str(Path(self.workspace.directory, self.input_file_grp)),
            model_path=model_path,
            resize_or_crop=self.parameter['imgresize'],
            loadSize=self.parameter['resizeHeight'],
            fineSize=self.parameter['resizeWidth'],
        )

        oplevel = self.parameter['operation_level']
        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %s", page_id)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()

            try:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page, page_id, feature_filter='dewarped', feature_selector='binarized')  # images should be deskewed and cropped
            except Exception:
                page_image, page_xywh, _ = self.workspace.image_from_page(
                    page, page_id, feature_filter='dewarped')  # images should be deskewed and cropped
            if oplevel == 'page':
                dataset = prepare_data(opt, page_image)
                orig_img_size = page_image.size
                self._process_segment(
                    model, dataset, page, page_xywh, page_id, input_file, orig_img_size, n)
            else:
                regions = page.get_TextRegion() + page.get_TableRegion()  # get all regions?
                if not regions:
                    LOG.warning("Page '%s' contains no text regions", page_id)
                for _, region in enumerate(regions):
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh)
                    # TODO: not tested on regions
                    # TODO: region has to exist as a physical file to be processed by pix2pixHD
                    dataset = prepare_data(opt, region_image)
                    orig_img_size = region_image.size
                    self._process_segment(
                        model, dataset, page, region_xywh, region.id, input_file, orig_img_size, n)

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

    def _process_segment(self, model, dataset, page, page_xywh, page_id, input_file, orig_img_size, n):
        for _, data in enumerate(dataset):
            w, h = orig_img_size
            generated = self.model.inference(data['label'], data['inst'], data['image'])
            #dewarped = generated.data[0].permute(1, 2, 0).detach().cpu().numpy()
            ## convert RGB float to uint8 (clipping negative)
            #dewarped = Image.fromarray(np.array(np.maximum(0, dewarped) * 255, dtype=np.uint8))
            # zzz: strictly, we should try to invert the dataset's input transform here
            dewarped = Image.fromarray(tensor2im(generated.data[0]))
            # resize using high-quality interpolation
            dewarped = dewarped.resize((w, h), Image.BICUBIC)
            # re-binarize
            dewarped = np.array(dewarped)
            dewarped = np.mean(dewarped, axis=2) > ocrolib.midrange(dewarped)
            dewarped = Image.fromarray(dewarped)
            coords['features'] += ',dewarped'
            file_id = make_file_id(input_file, self.output_file_grp) + '.IMG-DEW'
            file_path = self.workspace.save_image_file(dewarped,
                                                       file_id,
                                                       page_id=input_file.pageId,
                                                       file_grp=self.output_file_grp,
                                                      )
            page.add_AlternativeImage(AlternativeImageType(
                filename=file_path, comments=page_xywh['features']))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDewarper, *args, **kwargs)
