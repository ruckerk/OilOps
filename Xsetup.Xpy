import sys
from setuptools import setup
from OilOps import __version__

setup(
    name='OilOps',
    version = __version__,
    url='https://github.com/ruckerk/OilOps',
    author='W Kurt Rucker',
    author_email='OilOpsDev@gmail.com',
    packages=['OilOps'],
    install_requires=['adjustText',
                      'beautifulsoup4',
                      'datetime',
                      'easygui',
                      'futures3',
                      'geopy',
                      'lasio',
                      'matplotlib',
                      'multiprocess',
                      'numpy',
                      'openpyxl',
                      'pandas',
                      'psutil',
                      'pycrs',
                      'pyproj',
                      'pyshp',
                      'python-dateutil',
                      'python-magic-bin; sys.platform == "win32"',
                      'requests',
                      'scipy',
                      'selenium',
                      'scikit-learn',
                      'sqlalchemy',
                      'tabula-py',
                      'textract',
                      'urllib3',
                      'wget',
                      'xlrd', 
                      'geopandas'],
    license='MIT',
    description='Oilfield Operations Tools',
    long_description=open('README.txt').read(),
    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Typing :: Typed'
    ],
    keywords=[
        'OilOps', 'Colorado data', 'petrophysics', 'well spacing',
        'oilfield data', 'oilfield analysis', 'well survey', 'subsurface mapping'
    ],
)

#if "LINUX" in sys.platform.upper():
#    os.system('python magic not installed but required')
#    os.system('visit https://pypi.org/project/python-magic/ for installation detail')
#    os.system('sudo apt-get install libmagic1')
     
