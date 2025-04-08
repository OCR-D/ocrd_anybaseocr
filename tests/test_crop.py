# pylint: disable=import-error, unused-import, missing-docstring
from pathlib import Path

from ocrd import Resolver, Workspace, Resolver
from ocrd.processor.base import run_processor
from ocrd_utils import MIMETYPE_PAGE

from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import OcrdAnybaseocrCropper

from .assets import assets, copy_of_directory


def test_crop():
    resolver = Resolver()
    with copy_of_directory(assets.path_to('dfki-testdata/data')) as wsdir:
        ws = Workspace(resolver, wsdir)
        pagexml_before = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
        run_processor(
            OcrdAnybaseocrCropper,
            resolver=resolver,
            mets_url=str(Path(wsdir, 'mets.xml')),
            input_file_grp='BIN',
            output_file_grp='CROP-TEST',
            parameter={},
        )
        ws.reload_mets()
        pagexml_after = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
        assert pagexml_after == pagexml_before + 1, '1 file added'
        # assert next(ws.mets.find_files(mimetype=MIMETYPE_PAGE, fileGrp='CROP-TEST'))
