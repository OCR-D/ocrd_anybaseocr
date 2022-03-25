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
    # we cannot use TestOptions instances directly, because its parse()
    # does some nontrivial postprocessing (which we do not want to redo here)
    args = []
    args.extend(['--gpu_ids', str(gpu_id)])
    args.extend(['--nThreads', str(1)])   # test code only supports nThreads = 1
    args.extend(['--batchSize', str(1)])  # test code only supports batchSize = 1
    args.extend(['--serial_batches'])  # no shuffle
    args.extend(['--no_flip'])  # no flip
    args.extend(['--checkpoints_dir', str(model_path.parents[1])])
    args.extend(['--name', model_path.parents[0].name])
    args.extend(['--label_nc', str(0)]) # number of input label channels (just RGB if zero)
    args.extend(['--no_instance']) # no instance maps as input
    args.extend(['--resize_or_crop', resize_or_crop])
    args.extend(['--n_blocks_global', str(10)])
    args.extend(['--n_local_enhancers', str(2)])
    args.extend(['--loadSize', str(loadSize)])
    args.extend(['--fineSize', str(fineSize)])
    args.extend(['--model', 'pix2pixHD'])
    #args.extend(['--verbose'])
    LOG.debug("Options passed to pix2pixHD: %s", args)
    opt = TestOptions()
    opt = opt.parse(args=args, save=False, silent=True)

    model = create_model(opt)

    return opt, model

class OcrdAnybaseocrDewarper(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrDewarper, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp') and hasattr(self, 'parameter'):
            # processing context
            self.setup()

    def setup(self):
        LOG = getLogger('OcrdAnybaseocrDewarper')
        if self.parameter['gpu_id'] > -1 and not torch.cuda.is_available():
            LOG.warning("torch cannot detect CUDA installation.")
            self.parameter['gpu_id'] = -1

        model_path = Path(self.resolve_resource(self.parameter['model_path']))
        if not model_path.is_file():
            LOG.error("pix2pixHD model file was not found at '%s'", model_path)
            sys.exit(1)
        self.opt, self.model = prepare_options(
            gpu_id=self.parameter['gpu_id'],
            model_path=model_path,
            resize_or_crop=self.parameter['resize_mode'],
            loadSize=self.parameter['resize_height'],
            fineSize=self.parameter['resize_width'],
        )

    def process(self):
        """Dewarp pages of the workspace via pix2pixHD (conditional GANs)

        Open and deserialise each PAGE input file and its respective image,
        then iterate over its segment hierarchy down to the requested
        ``operation_level``.

        Next, get the binarized image according to the layout annotation
        (from the alternative image of the segment, or by cropping and
        deskewing from the parent image as annotated).

        Then pass the image to the preloaded pix2pixHD model for inference.
        (It will be resized and/or cropped according to ``resize_width``,
        ``resize_height`` and ``resize_mode`` prior to decoding, and the
        result will be resized to match the original.)

        After decoding, add the new image file to the output fileGrp for
        the same pageId (using a file ID with suffix ``.IMG-DEW``).
        Reference the new image file in the AlternativeImage of the segment.

        Produce a new output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('OcrdAnybaseocrDewarper')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        oplevel = self.parameter['operation_level']
        for input_file in self.input_files:
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %s", page_id)

            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()

            page_image, page_xywh, _ = self.workspace.image_from_page(
                page, page_id,
                # images SHOULD be deskewed and cropped, and MUST be binarized
                feature_filter='dewarped', feature_selector='binarized')
            if oplevel == 'page':
                self._process_segment(
                    prepare_data(self.opt, page_image), page, page_xywh, page_image.size, input_file)
            else:
                regions = page.get_TextRegion() + page.get_TableRegion()  # get all regions?
                if not regions:
                    LOG.warning("Page '%s' contains no text regions", page_id)
                for _, region in enumerate(regions):
                    region_image, region_xywh = self.workspace.image_from_segment(
                        region, page_image, page_xywh,
                        # images SHOULD be deskewed and cropped, and MUST be binarized
                        feature_filter='dewarped', feature_selector='binarized')
                    # TODO: not tested on regions
                    self._process_segment(
                        prepare_data(self.opt, region_image), region, region_xywh, region_image.size, input_file)

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

    def _process_segment(self, dataset, segment, coords, orig_img_size, input_file):
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
            file_id = make_file_id(input_file, self.output_file_grp) + '_' + segment.id + '.IMG-DEW'
            file_path = self.workspace.save_image_file(dewarped,
                                                       file_id,
                                                       page_id=input_file.pageId,
                                                       file_grp=self.output_file_grp,
            )
            segment.add_AlternativeImage(AlternativeImageType(
                filename=file_path, comments=coords['features']))

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDewarper, *args, **kwargs)
