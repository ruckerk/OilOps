# OilOps

OilOps is a project bringing together multiple tools for oilfield operations

UWI is an API manager than easily converts API# from numeric or dashed strings to integers or dashed strings for API number lengths
CO folder tools are for collecting Colorado public well data
LAS folder is for petrophysics
PRODUCTION folder is for production analysis
SURVEY folder contains well spacing tools
MAPS folder manipulates shapefiles and maps
DATA folder contains data acquisition processes

pip install -I --no-deps  git+https://github.com/ruckerk/OilOps#egg=OilOps

Windows hangup:
libmagic.dll installed to \Lib\site-packages\magic\libmagic but needs to be in a directory in %PATH%
Solutions:
    Copy libraries from site-packages/magic/libmagic to current folder and magic will find them.
    Add site-packages/magic/libmagic to PATH.
Linux hangup:
    need to sudo install libmagic
    https://pypi.org/project/python-magic/ for installation detail

