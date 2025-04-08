import json

from .base import main, CapturingTestCase as TestCase

from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import cli as OcrdAnybaseocrCropper

CLIS = [
        OcrdAnybaseocrCropper,
]

class SmokeTest(TestCase):

    def test_all_help(self):
        """
        Make sure all CLIs produce --help output
        """
        for cli in CLIS:
            exit_code, out, err = self.invoke_cli(cli, ['--help'])
            self.assertIn('--input-file-grp', out)
            self.assertEqual(exit_code, 0)

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
