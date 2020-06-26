# pylint: disable=import-error, unused-import, missing-docstring
from pathlib import Path

from tests.base import TestCase, assets, main, copy_of_directory
from ocrd_anybaseocr.cli.ocrd_anybaseocr_dewarp import OcrdAnybaseocrDewarper # FIXME srsly y
from ocrd import Resolver
from ocrd.processor.base import run_processor


class AnyocrDewarperTest(TestCase):

    def setUp(self):
        self.model_path = Path(Path.cwd(), 'models/latest_net_G.pth')

    def test_dewarp(self):
        with copy_of_directory(assets.path_to('dfki-testdata/data')) as wsdir:
            run_processor(
                OcrdAnybaseocrDewarper,
                resolver=Resolver(),
                mets_url=str(Path(wsdir, 'mets.xml')),
                input_file_grp='OCR-D-IMG-BIN',
                output_file_grp='DEWARP',
                parameter={'model_path': str(self.model_path)},
            )

if __name__ == "__main__":
    main(__file__)
