import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_anybaseocr.cli.ocrd_anybaseocr_tiseg import OcrdAnybaseocrTiseg
from ocrd_anybaseocr.cli.ocrd_anybaseocr_textline import OcrdAnybaseocrTextline


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_tiseg(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrTiseg, *args, **kwargs)


@click.command()
@ocrd_cli_options
def ocrd_anybaseocr_textline(*args, **kwargs):
    return ocrd_cli_wrap_processor(OcrdAnybaseocrTextline, *args, **kwargs)
