# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import json
with open('ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']

setup(
    name='ocrd-anybaseocr',
    version=version,
    author="DFKI",
    author_email="Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de",
    url="https://github.com/OCR-D/ocrd_anybaseocr",
    license='Apache License 2.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().split('\n'),
    packages=find_packages(exclude=["work_dir", "src"]),
    package_data={
        '': ['*.json']
    },
    entry_points={
        'console_scripts': [
            'ocrd-anybaseocr-binarize           = ocrd_anybaseocr.cli.ocrd_anybaseocr_binarize:cli',
            'ocrd-anybaseocr-deskew             = ocrd_anybaseocr.cli.ocrd_anybaseocr_deskew:cli',
            'ocrd-anybaseocr-crop               = ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping:cli',
            'ocrd-anybaseocr-dewarp             = ocrd_anybaseocr.cli.ocrd_anybaseocr_dewarp:cli',
            'ocrd-anybaseocr-tiseg              = ocrd_anybaseocr.cli.ocrd_anybaseocr_tiseg:cli',
            'ocrd-anybaseocr-textline           = ocrd_anybaseocr.cli.ocrd_anybaseocr_textline:cli',
            'ocrd-anybaseocr-layout-analysis    = ocrd_anybaseocr.cli.ocrd_anybaseocr_layout_analysis:cli',
            'ocrd-anybaseocr-block-segmentation = ocrd_anybaseocr.cli.ocrd_anybaseocr_block_segmentation:cli'
        ]
    },
)
