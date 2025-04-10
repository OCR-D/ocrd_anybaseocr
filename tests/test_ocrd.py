# pylint: disable=import-error, unused-import, missing-docstring
import os
from pathlib import Path

from ocrd import Resolver, Workspace, Resolver
from ocrd.processor.base import run_processor
from ocrd_utils import MIMETYPE_PAGE

from ocrd_anybaseocr.cli.ocrd_anybaseocr_cropping import OcrdAnybaseocrCropper
from ocrd_anybaseocr.cli.ocrd_anybaseocr_layout_analysis import OcrdAnybaseocrLayoutAnalyser


def test_crop(processor_kwargs):
    ws  = processor_kwargs['workspace']
    pagexml_before = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
    run_processor(
        OcrdAnybaseocrCropper,
        input_file_grp='BIN',
        output_file_grp='CROP-TEST',
        parameter={},
        **processor_kwargs
    )
    ws.reload_mets()
    pagexml_after = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
    assert pagexml_after == pagexml_before + 1, '1 file added'

def test_layout(processor_kwargs):
    ws  = processor_kwargs['workspace']
    readable_workspace = Workspace(ws.resolver, ws.directory,
                                    mets_basename=os.path.basename(ws.mets_target))
    mets_xml_before = readable_workspace.mets.to_xml(xmllint=True).decode('utf-8')
    assert 'mets:structMap TYPE="LOGICAL"' not in mets_xml_before, mets_xml_before
    run_processor(
        OcrdAnybaseocrLayoutAnalyser,
        input_file_grp='CROP',
        output_file_grp='LAYOUT',
        parameter={},
        **processor_kwargs
    )
    ws.save_mets()
    readable_workspace.reload_mets()
    mets_xml_after = readable_workspace.mets.to_xml(xmllint=True).decode('utf-8')
    assert len(mets_xml_after) > len(mets_xml_before), (mets_xml_before, mets_xml_after)
    assert 'mets:structMap TYPE="LOGICAL"' in mets_xml_after, mets_xml_after
