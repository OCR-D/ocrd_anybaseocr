import json
from pkg_resources import resource_string

__all__ = ['OCRD_TOOL']

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))
