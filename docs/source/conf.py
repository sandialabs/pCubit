import os
import sys
import re
sys.path.insert(0, os.path.abspath("../../"))


def get_version():
    VERSIONFILE = os.path.join('..', '..', 'pyproject.toml')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
        vgx = 'version = \'[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\''
        for line in lines:
            mo = re.search(vgx, line, re.M)
            if mo:
                return mo.group().split("'")[1]
        raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

project = 'pCubit'
copyright = '2024, National Technology & Engineering Solutions of Sandia, LLC.'
author = 'Michael R. Buche'
release = get_version()

extensions = []
exclude_patterns = []
html_show_sphinx = False
html_show_sourcelink = False
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
templates_path = ['_templates']
