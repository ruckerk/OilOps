# update base files
import wget
from zipfile import ZipFile
import os,sys, shutil
from glob import glob

pathname = os.path.dirname(sys.argv[0])
adir = os.path.abspath(pathname)

#Frac Focus
#https://www.fracfocus.org/index.php?p=data-download
url = 'https://fracfocusdata.org/digitaldownload/FracFocusCSV.zip'
filename = wget.download(url)
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall('FRAC_FOCUS')


# COGCC SQLITE
# https://dnrftp.state.co.us/
url = 'https://dnrftp.state.co.us/COGCC/Temp/Gateway/CO_3_2.1.zip'
filename = wget.download(url)
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall('COOGC_SQL')

files = []
start_dir = os.path.join(adir,'COOGC_SQL')
pattern   = r'CO_3_2.*'

for dir,_,_ in os.walk(start_dir):
    files.extend(glob(os.path.join(dir,pattern)))

shutil.move(files[0], os.path.join(adir, os.path.basename(files[0])))
shutil.rmtree(path.join(adir,'COOGC_SQL'))

# COGCC shapefiles
url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_SHP.ZIP'
filename = wget.download(url)
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
os.remove(filename)

url = 'https://cogcc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_PENDING_SHP.ZIP'
filename = wget.download(url)
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
os.remove(filename)

url = 'https://cogcc.state.co.us/documents/data/downloads/gis/WELLS_SHP.ZIP'
filename = wget.download(url)
with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
os.remove(filename)
