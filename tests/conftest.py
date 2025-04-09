# pylint: disable=unused-import

from multiprocessing import Process
from time import sleep
from random import seed, sample
import logging
import os
import pytest

from ocrd import Resolver, Workspace, OcrdMetsServer
from ocrd_utils import pushd_popd, disableLogging, initLogging, setOverrideLogLevel, config

from .assets import assets

DEFAULT_WS = "dfki-testdata"
WORKSPACES = {
    DEFAULT_WS: assets.path_to(os.path.join(DEFAULT_WS, 'data', 'mets.xml')),
}

#@pytest.fixture(params=WORKSPACES.keys())
@pytest.fixture
def workspace(tmpdir, pytestconfig, asset):
    initLogging()
    logging.getLogger('ocrd.processor').setLevel(logging.DEBUG)
    #if pytestconfig.getoption('verbose') > 0:
    #    setOverrideLogLevel('DEBUG')
    config.OCRD_MISSING_OUTPUT = "ABORT"
    with pushd_popd(tmpdir):
        directory = str(tmpdir)
        resolver = Resolver()
        url = WORKSPACES[asset]
        workspace = resolver.workspace_from_url(url, dst_dir=directory) # download=True
        workspace.name = asset # for debugging
        # download only up to 4 pages
        pages = workspace.mets.physical_pages
        if len(pages) > 4:
            seed(12) # make tests repeatable
            pages = sample(pages, 4)
        page_id = ','.join(pages)
        for file in workspace.find_files(page_id=page_id):
            if file.url.startswith("file:/") or file.fileGrp in ["THUMBS", "MIN"]:
                # ignore broken and irrelevant groups
                # (first image group will be used for alto_processor tests)
                workspace.remove_file(file.ID, force=True)
            else:
                workspace.download_file(file)
        yield workspace, page_id
    config.reset_defaults()
    disableLogging()

def pytest_addoption(parser):
    parser.addoption("--workspace",
                     action="append",
                     choices=list(WORKSPACES) + ["all"],
                     help="workspace(s) to run on (set 'all' to use all)"
    )

@pytest.hookimpl
def pytest_generate_tests(metafunc):
    if "asset" in metafunc.fixturenames:
        ws = metafunc.config.getoption("workspace")
        if ws == ['all']:
            ws = list(WORKSPACES)
        elif not ws:
            ws = [DEFAULT_WS]
        metafunc.parametrize("asset", ws)

CONFIGS = ['', 'pageparallel', 'metscache', 'pageparallel+metscache']

@pytest.fixture(params=CONFIGS)
def processor_kwargs(request, workspace):
    config.OCRD_DOWNLOAD_INPUT = False # only 4 pre-downloaded pages
    workspace, page_id = workspace
    config.OCRD_MISSING_OUTPUT = "ABORT"
    if 'metscache' in request.param:
        config.OCRD_METS_CACHING = True
        #print("enabled METS caching")
    if 'pageparallel' in request.param:
        config.OCRD_MAX_PARALLEL_PAGES = 4
        #print("enabled page-parallel processing")
        def _start_mets_server(*args, **kwargs):
            #print("running with METS server")
            server = OcrdMetsServer(*args, **kwargs)
            server.startup()
        process = Process(target=_start_mets_server,
                          kwargs={'workspace': workspace, 'url': 'mets.sock'})
        process.start()
        sleep(1)
        # instantiate client-side workspace
        asset = workspace.name
        workspace = Workspace(workspace.resolver, workspace.directory,
                              mets_server_url='mets.sock',
                              mets_basename=os.path.basename(workspace.mets_target))
        workspace.name = asset
        yield {'workspace': workspace, 'page_id': page_id, 'mets_server_url': 'mets.sock'}
        process.terminate()
    else:
        yield {'workspace': workspace, 'page_id': page_id}
    config.reset_defaults()
