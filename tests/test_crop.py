# pylint: disable=import-error, unused-import, missing-docstring
from pathlib import Path

from ocrd import Resolver, Workspace
from ocrd.processor.base import run_processor
from ocrd_utils import MIMETYPE_PAGE, initLogging

from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import OcrdAnybaseocrCropper

from .base import TestCase, assets, main, copy_of_directory


class AnyocrCropperTest(TestCase):

    def setUp(self):
        self.resolver = Resolver()
        initLogging()

    def test_crop(self):
        with copy_of_directory(assets.path_to('dfki-testdata/data')) as wsdir:
            ws = Workspace(self.resolver, wsdir)
            pagexml_before = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
            run_processor(
                OcrdAnybaseocrCropper,
                resolver=self.resolver,
                mets_url=str(Path(wsdir, 'mets.xml')),
                input_file_grp='BIN',
                output_file_grp='CROP-TEST',
                parameter={},
            )
            ws.reload_mets()
            pagexml_after = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
            self.assertEqual(pagexml_after, pagexml_before + 1)

if __name__ == "__main__":
    main(__file__)

