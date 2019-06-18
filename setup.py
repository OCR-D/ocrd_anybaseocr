# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = 'layout-analysis',
    version = 'v0.0.1',
    author = "Syed Saqib Bukhari",
    author_email = "Saqib.Bukhari@dfki.de, Mohammad_mohsin.reza@dfki.de",
    url = "https://github.com/mjenckel/LAYoutERkennung",
    license='Apache License 2.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().split('\n'),
    packages=find_packages(exclude=["work_dir","src"]),
    package_data={
        '': ['*.json']
    },
    entry_points={
        'console_scripts': [
            'ocrd-anybaseocr-tiseg = ocrd_anybaseocr.cli.ocrd_anybaseocr_tiseg:main',
            'ocrd-anybaseocr-textline = ocrd_anybaseocr.cli.ocrd_anybaseocr_textline:main'            
        ]
    },
)