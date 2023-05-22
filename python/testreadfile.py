import os.path
# import cupy
# import platform
import datetime as dt
# from aiapy.calibrate.prep import correct_degradation
import numpy as np
import glob
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import matplotlib.colors as colors
import astropy.units as u
from astropy.coordinates import SkyCoord
# import astropy.time as time
from astropy.io import fits as fits
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
# from sunpy.coordinates import propagate_with_solar_surface
# import scipy.io as io
# from dn2dem_pos import dn2dem_pos
import warnings
warnings.simplefilter('ignore')
# from aiapy.calibrate import register, update_pointing, estimate_error, normalize_exposure
# import aiapy.psf
# import asdf
from bisect import bisect_left, bisect_right





def closest(list, value):
    """
    Get the closest element to value in list

    Parameters
    ----------
    list: Numpy array
        Array of numbers in which to find the value
    value: Number
        Number to find in list
    
    Return: Integer of the number in list closest to value
    """
    ind = np.abs([elem - value for elem in list])
    return ind.argmin(0)

## Function to get the list of AIA filenames
def get_filelist(data_disk, passband, img_file_date, img_time_range):
    files = glob.glob(data_disk+str(passband)+'/*.fits', recursive=True)
    files.sort()
    files_dt = []
    for file_i in files:
        try:
            hdr = fits.getheader(file_i, 1, ignore_missing_simple=True)
            try:
                files_dt.append(dt.datetime.strptime(hdr.get('DATE-OBS'),'%Y-%m-%dT%H:%M:%S.%fZ'))
            except:
                files_dt.append(dt.datetime.strptime(hdr.get('DATE-OBS'),'%Y-%m-%dT%H:%M:%S.%f'))    
        except OSError as e:
            # Download new file
            print('{}'.format(e) + ' for' + file_i +': Downloading new file')
            files_dtload = dt.datetime.strptime(file_i.split(f'{passband}a_')[1].split('z')[0], '%Y_%m_%dt%H_%M_%S_%f')
            file_load = get_data(files_dtload-dt.timedelta(seconds=10), files_dtload+dt.timedelta(seconds=10), img_file_date, 10*u.second, passband, data_disk)
            print("Download complete")
            files_dt.append(files_dtload)
            # hdr = fits.getheader(file_load, 1, ignore_missing_simple=True)
            # try:
            #     files_dt.append(dt.datetime.strptime(hdr.get('DATE-OBS'),'%Y-%m-%dT%H:%M:%S.%fZ'))
            # except:
            #     files_dt.append(dt.datetime.strptime(hdr.get('DATE-OBS'),'%Y-%m-%dT%H:%M:%S.%f'))  

    
    left = bisect_left(files_dt, img_time_range[0])
    right = bisect_right(files_dt, img_time_range[1])
    files_out = files[left:right]
    file_time_out = files_dt[left:right]

    return files_out, file_time_out

## Function with event information
def event_info(data_disk):
    start_time = '2018/10/31 00:01:23'
    end_time = '2018/10/31 01:00:00'
    cadence = 10*u.second #seconds
    img_time_range = [dt.datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S"), dt.datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")]

    ref_time = '2018/10/31 00:00:09'
    # bottom_left = [1637, 379]*u.pixel  
    # top_right = [2889, 1630]*u.pixel  

    ref_file_date = dt.datetime.strftime(dt.datetime.strptime(ref_time,'%Y/%m/%d %H:%M:%S'), '%Y/%m/%d')
    img_file_date = dt.datetime.strftime(dt.datetime.strptime(ref_time,'%Y/%m/%d %H:%M:%S'), '%Y/%m/%d')

    strt_time = dt.datetime.strptime(ref_time, "%Y/%m/%d %H:%M:%S")
    ref_time_range = [strt_time-dt.timedelta(seconds=10), strt_time+dt.timedelta(seconds=10)]

    files, files_dt = get_filelist(data_disk, 193, ref_file_date, ref_time_range)
    if not files:
        get_data(strt_time-dt.timedelta(seconds=10), strt_time+dt.timedelta(seconds=10), ref_file_date, 10*u.second, 193, data_disk)

    # files, files_dt = get_filelist(data_disk, 193, ref_file_date, ref_time_range)

    ind = np.abs([t - strt_time for t in files_dt])
    map = Map(files[ind.argmin()])
    bottom_left = SkyCoord(-650 * u.arcsec, -600 * u.arcsec, frame= map.coordinate_frame)
    bottom_left_pix = skycoord_to_pixel(bottom_left, map.wcs, origin = 0)*u.pixel
    top_right = SkyCoord(100 * u.arcsec, 600 * u.arcsec, frame= map.coordinate_frame)
    top_right_pix = skycoord_to_pixel(top_right, map.wcs, origin = 0)*u.pixel
    
    pix_width = [(top_right_pix[0]-bottom_left_pix[0])/2, (top_right_pix[1]-bottom_left_pix[1])/2]
    pix_centre = [pix_width[0]+bottom_left_pix[0], pix_width[1]+bottom_left_pix[1]]
    crd_bl = SkyCoord(map.pixel_to_world(bottom_left_pix[0],bottom_left_pix[1]),frame = map.coordinate_frame)
    crd_tr = SkyCoord(map.pixel_to_world(top_right_pix[0],top_right_pix[1]),frame = map.coordinate_frame)
    
    crd_cent = SkyCoord(map.pixel_to_world(pix_centre[0],pix_centre[1]),frame = map.coordinate_frame)
    crd_width = [(crd_tr.Tx.arcsecond-crd_bl.Tx.arcsecond)/2, (crd_tr.Ty.arcsecond-crd_bl.Ty.arcsecond)/2]
    
    return start_time, end_time, ref_time, cadence, crd_cent, crd_width, ref_file_date, img_file_date, img_time_range



## Function to download data
def get_data(start_time, end_time, img_file_date, cadence, pband, data_disk):
    ## Identify and download the data
    attrs_time = a.Time(start_time, end_time)
    wvlnth = a.Wavelength(int(pband)*u.Angstrom, int(pband)*u.Angstrom)
    result = Fido.search(attrs_time, a.Instrument('AIA'), wvlnth, a.Sample(cadence))
    files = Fido.fetch(result, path = data_disk+str(pband)+'/', overwrite=True, progress=True)
    

data_disk = '/disk/solar/nn2/data/2018/10/31/00/AIA'

# os.makedirs(data_disk, exist_ok='True')

## Get the event information
start_time,end_time,ref_time,cadence,crd_cent,crd_width,ref_file_date,img_file_date,img_time_range = event_info(data_disk)

## Define and create the output directories

output_dir = '/disk/solar/nn2/results/DEM_update/'

os.makedirs(output_dir, exist_ok='True')
passband = [94, 131, 171, 193, 211, 335]

## Download the data
# for pband in passband:
#     files, file_time = get_filelist(data_disk, pband, img_file_date, img_time_range)
#     n_img = ((img_time_range[1]-img_time_range[0]).total_seconds()/(cadence/u.second))

#     if n_img > len(files):
#         print('Fewer than expected FITS files for '+str(pband)+' passband')
#         print('Downloading data for '+str(pband)+' passband')
#         get_data(start_time, end_time, img_file_date, cadence, pband, data_disk)
#     else:
#         print('Data already downloaded for '+str(pband).rjust(4, "0")+' passband')

## Get list of files from each passband to identify the smallest number of files
print('Getting list of files')
f_0094, time_0094 = get_filelist(data_disk, 94, img_file_date, img_time_range)
f_0131, time_0131 = get_filelist(data_disk, 131, img_file_date, img_time_range)
f_0171, time_0171 = get_filelist(data_disk, 171, img_file_date, img_time_range)
f_0193, time_0193 = get_filelist(data_disk, 193, img_file_date, img_time_range)
f_0211, time_0211 = get_filelist(data_disk, 211, img_file_date, img_time_range)
f_0335, time_0335 = get_filelist(data_disk, 335, img_file_date, img_time_range)

flength = [len(f_0094), len(f_0131), len(f_0171), len(f_0193), len(f_0211), len(f_0335)]
flist = [f_0094, f_0131, f_0171, f_0193, f_0211, f_0335]
time_array = [time_0094, time_0131, time_0171, time_0193, time_0211, time_0335]
index = np.argmin(flength)

print(flength)