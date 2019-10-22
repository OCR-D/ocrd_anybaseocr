import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_anybaseocr.cli.ocrd_anybaseocr_binarize import OcrdAnybaseocrBinarizer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_deskew import OcrdAnybaseocrDeskewer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import OcrdAnybaseocrCropper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_dewarp import OcrdAnybaseocrDewarper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_tiseg import OcrdAnybaseocrTiseg
from ocrd_anybaseocr.cli.ocrd_anybaseocr_textline import OcrdAnybaseocrTextline
from ocrd_anybaseocr.cli.ocrd_anybaseocr_layout_analysis import OcrdAnybaseocrLayoutAnalyser
from ocrd_anybaseocr.cli.ocrd_anybaseocr_block_segmentation import OcrdAnybaseocrBlockSegmenter


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_binarize(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrBinarizer, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_deskew(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDeskewer, *args, **kwargs)    

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_cropping(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrCropper, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_dewarp(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrDewarper, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_tiseg(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrTiseg, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_textline(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrTextline, *args, **kwargs)


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_layout_analysis(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrLayoutAnalyser, *args, **kwargs)


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_block_segmentation(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrBlockSegmenter, *args, **kwargs)

