import json

from tests.base import main, CapturingTestCase as TestCase

from ocrd_anybaseocr.cli.ocrd_anybaseocr_binarize import cli as OcrdAnybaseocrBinarizer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_block_segmentation import cli as OcrdAnybaseocrBlockSegmenter
from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import cli as OcrdAnybaseocrCropper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_deskew import cli as OcrdAnybaseocrDeskewer
from ocrd_anybaseocr.cli.ocrd_anybaseocr_dewarp import cli as OcrdAnybaseocrDewarper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_layout_analysis import cli as OcrdAnybaseocrLayoutAnalyser
from ocrd_anybaseocr.cli.ocrd_anybaseocr_textline import cli as OcrdAnybaseocrTextline
from ocrd_anybaseocr.cli.ocrd_anybaseocr_tiseg import cli as OcrdAnybaseocrTiseg

CLIS = [
        OcrdAnybaseocrBinarizer,
        OcrdAnybaseocrBlockSegmenter,
        OcrdAnybaseocrCropper,
        OcrdAnybaseocrDeskewer,
        OcrdAnybaseocrDewarper,
        OcrdAnybaseocrLayoutAnalyser,
        OcrdAnybaseocrTextline,
        OcrdAnybaseocrTiseg
]

class SmokeTest(TestCase):

    def test_all_help(self):
        """
        Make sure all CLIs produce --help output
        """
        for cli in CLIS:
            exit_code, out, err = self.invoke_cli(cli, ['--help'])
            self.assertIn('--input-file-grp', out)
            self.assertEquals(exit_code, 0)

    def test_all_json(self):
        """
        Make sure all CLIs produce --dump-json output on stdout
        """
        for cli in CLIS:
            exit_code, out, err = self.invoke_cli(cli, ['--dump-json'])
            parsed = json.loads(out)
            self.assertTrue(parsed['description'])

if __name__ == '__main__':
    main(__file__)
