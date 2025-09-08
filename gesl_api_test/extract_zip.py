from zipfile import ZipFile
import glob as glob

sigid_arr = glob.glob("sigid-*.zip")
for fname in sigid_arr:
    with ZipFile(fname, 'r') as zip:
        # zip.printdir()
        zip.extractall()

# raw_arr = glob.glob("GridSweep-*.zip")
# for fname_raw in raw_arr:
#     with ZipFile(fname, 'r') as zip:
#         # zip.printdir()
#         zip.extractall()
