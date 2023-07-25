# Routine to get DEM from AIA data using Hannah & Kontar implementation

import os.path
import cupy
import platform
import datetime as dt
from aiapy.calibrate.prep import correct_degradation
import numpy as np
import glob
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.time as time
from astropy.io import fits as fits
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from sunpy.map import Map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
import scipy.io as io
from dn2dem_pos import dn2dem_pos
import warnings
warnings.simplefilter('ignore')
from aiapy.calibrate import register, update_pointing, estimate_error, normalize_exposure
import aiapy.psf
import asdf
from bisect import bisect_left, bisect_right
from multiprocessing import set_start_method



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


# Function to flatten list
def flatten(l):
    return [item for sublist in l for item in sublist]

## Function to get the list of AIA filenames and time, Default is for img_time_range.
def get_filelist_AIA(data_disk, passband, img_file_date, img_time_range):
    print('Getting list of files for AIA'+str(passband)+' passband')
    hstart = img_time_range[0].hour
    hend = img_time_range[1].hour
    if hstart == hend:
        h = np.arange(hstart,hend+1,1)
    else:
        h = np.arange(hstart,hend,1)
    
    files_list = []
    files_dt_list = []
    for i in h:
        print('Looking for hour '+str(i))
        files = glob.glob(data_disk+img_file_date+'/'+str(i).zfill(2)+'/AIA'+str(passband)+'/*.fits', recursive=True)
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
                file_load = get_data_AIA(files_dtload-dt.timedelta(seconds=10), files_dtload+dt.timedelta(seconds=10), 10*u.second, passband, data_disk+img_file_date, i)
                print("Download complete")
                files_dt.append(files_dtload)
        files_list.append(files)
        files_dt_list.append(files_dt)
    
    files_dt_list = flatten(files_dt_list)
    files_list = flatten(files_list)

    left = bisect_left(files_dt_list, img_time_range[0])
    right = bisect_right(files_dt_list, img_time_range[1])
    files_out = files_list[left:right]
    file_time_out = files_dt_list[left:right]
    
    return files_out, file_time_out

## Function to download data (default instrument is AIA)
def get_data_AIA(start_time, end_time, cadence, pband, data_disk, hour, overwrite = True):
    ## Identify and download the data
    attrs_time = a.Time(start_time, end_time)
    wvlnth = a.Wavelength(int(pband)*u.Angstrom, int(pband)*u.Angstrom)
    result = Fido.search(attrs_time, a.Instrument('AIA'), wvlnth, a.Sample(cadence))
    files = Fido.fetch(result, path = data_disk+'/'+str(hour).zfill(2)+'/AIA'+str(pband)+'/', overwrite=overwrite, progress=True)


# Function with event information (1 day only) and define reference box
def event_info(data_disk, data_disk_date):
    # Exception for 28 Oct and 4 Nov
    if data_disk_date == '2018/10/28':
        start_time = data_disk_date+ ' 12:00:00'
        end_time = data_disk_date+ ' 23:59:59'
    elif data_disk_date == '2018/11/04':
        start_time = data_disk_date+ ' 00:00:00'
        end_time = data_disk_date+ ' 12:00:00'
    else:
        start_time = data_disk_date+ ' 00:00:00'
        end_time = data_disk_date+ ' 23:59:59'

    cadence = 10*u.minute #cadence of AIA images
    img_time_range = [dt.datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S"), dt.datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")]

    ref_time = '2018/10/31 00:00:09'
    # bottom_left = [1637, 379]*u.pixel  
    # top_right = [2889, 1630]*u.pixel  

    #Use hour instead of date to download ref
    ref_file_date = dt.datetime.strftime(dt.datetime.strptime(ref_time,'%Y/%m/%d %H:%M:%S'), '%Y/%m/%d')
    img_file_date = dt.datetime.strftime(dt.datetime.strptime(data_disk_date,'%Y/%m/%d'), '%Y/%m/%d')

    strt_time = dt.datetime.strptime(ref_time, "%Y/%m/%d %H:%M:%S")
    hour_ref = strt_time.hour
    ref_time_range = [strt_time-dt.timedelta(seconds=9), strt_time+dt.timedelta(seconds=10)]

    files, files_dt = get_filelist_AIA(data_disk, 193, ref_file_date, ref_time_range)

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

## Function to get AIA submap
def get_submap(time_array,index,img,file,crd_cent,crd_width):
    ind = closest(np.array(time_array[3][:]), time_array[index][img])
    map = Map(file[ind])
    with propagate_with_solar_surface():
        diffrot_cent = crd_cent.transform_to(map.coordinate_frame)
    bl = SkyCoord((diffrot_cent.Tx.arcsecond-crd_width[0])*u.arcsec, (diffrot_cent.Ty.arcsecond-crd_width[1])*u.arcsec,
                  frame = map.coordinate_frame)
    tr = SkyCoord((diffrot_cent.Tx.arcsecond+crd_width[0])*u.arcsec, (diffrot_cent.Ty.arcsecond+crd_width[1])*u.arcsec,
                  frame = map.coordinate_frame)
    submap = map.submap(bl, top_right=tr)
    return submap

## Function to prep AIA images, deconvolve with PSF and produce submap
def prep_images(time_array,index,img,f_0094,f_0131,f_0171,f_0193,f_0211,f_0335,crd_cent,crd_width):
    ind_0094 = closest(np.array(time_array[0][:]), time_array[index][img])
    ind_0131 = closest(np.array(time_array[1][:]), time_array[index][img])
    ind_0171 = closest(np.array(time_array[2][:]), time_array[index][img])
    ind_0193 = closest(np.array(time_array[3][:]), time_array[index][img])
    ind_0211 = closest(np.array(time_array[4][:]), time_array[index][img])
    ind_0335 = closest(np.array(time_array[5][:]), time_array[index][img])

    farray = [f_0094[ind_0094], f_0131[ind_0131], f_0171[ind_0171], f_0193[ind_0193], f_0211[ind_0211], f_0335[ind_0335]]
    maps = Map(farray)

    print('Prepping images & deconvolving with PSF')
    
    for m in range(0, len(maps)):
        psf = aiapy.psf.psf(maps[m].wavelength)
        aia_map_deconvolved = aiapy.psf.deconvolve(maps[m], psf=psf)
        aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
        aia_map_registered = register(aia_map_updated_pointing)
        aia_map_corrected = correct_degradation(aia_map_registered)
        aia_map_norm = normalize_exposure(aia_map_corrected)
        #Replace maps with prepped maps
        maps[m] = aia_map_norm

    #Diff rotate and get submap
    with propagate_with_solar_surface():
        diffrot_cent = crd_cent.transform_to(maps[3].coordinate_frame)
    bl = SkyCoord((diffrot_cent.Tx.arcsecond-crd_width[0])*u.arcsec, (diffrot_cent.Ty.arcsecond-crd_width[1])*u.arcsec,
                  frame = maps[3].coordinate_frame)
    bl_x, bl_y = maps[3].world_to_pixel(bl)
    tr = SkyCoord((diffrot_cent.Tx.arcsecond+crd_width[0])*u.arcsec, (diffrot_cent.Ty.arcsecond+crd_width[1])*u.arcsec,
                  frame = maps[0].coordinate_frame)
    tr_x, tr_y = maps[3].world_to_pixel(tr)
    submap_0 = maps[3].submap([int(bl_x.value), int(bl_y.value)]*u.pixel, top_right=[int(tr_x.value), int(tr_y.value)]*u.pixel)
    nx,ny = submap_0.data.shape
    nf=len(maps)
    map_arr = []
    error_array = np.zeros([nx, ny, nf])

    eclipse_correction = [0.7,0.6,3.4,5.2,1.6,0.6] #Heinemann et al. (2021)
    for n in range(0, len(maps)):
        submap = maps[n].submap([int(bl_x.value), int(bl_y.value)]*u.pixel, top_right=[int(tr_x.value), int(tr_y.value)]*u.pixel)
        map_arr.append(submap)
        num_pix=submap.data.size
        error_array[:,:,n] = estimate_error(submap.data*(u.ct/u.pix),submap.wavelength,num_pix)
    #Correct values of DEM estimated error - Using eclipse correction by Heinemann et al. (2021)
        error_array[:,:,n] = error_array[:,:,n]+eclipse_correction[n]

    map_array = Map(map_arr[0],map_arr[1],map_arr[2],map_arr[3],
                    map_arr[4],map_arr[5],sequence=True,sortby=None) 
    print('Images prepped & region of interest selected')

    return map_array, error_array
    
## Function to calculate DEM
def calculate_dem(map_array, err_array):
    nx,ny = map_array[0].data.shape
    nf=len(map_array)
    image_array = np.zeros((nx,ny,nf))
    for img in range(0,nf):
        image_array[:,:,img] = map_array[img].data
    # print('Created empty array for image data')

    trin=io.readsav('/disk/solar/nn2/demreg/python/aia_tresp_en.dat')
    # print('Loaded AIA response function')
        
    tresp_logt=np.array(trin['logt'])
    nt=len(tresp_logt)
    nf=len(trin['tr'][:])
    trmatrix=np.zeros((nt,nf))
    for i in range(0,nf):
        trmatrix[:,i]=trin['tr'][i]    
    
    t_space=0.1
    #logT = 5.3-6.5 (Heinemann2021) #lower lim 5.5
    t_min=5.5
    t_max=6.5
    logtemps=np.linspace(t_min,t_max,num=int((t_max-t_min)/t_space)+1)
    temps=10**logtemps
    mlogt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) for i in np.arange(0,len(temps)-1)])
    print('Start dn2dem_pos function (Ian Hannah & Eduard Kontar code)')
    dem,edem,elogt,chisq,dn_reg=dn2dem_pos(image_array,err_array,trmatrix,tresp_logt,temps,max_iter=15)
    dem = dem.clip(min=0)
    return dem,edem,elogt,chisq,dn_reg,mlogt,logtemps

## Function to plot the DEM images
def plot_dem_images(submap,dem,logtemps,img_arr_tit):
    nt=len(dem[0,0,:])
    # nt_new=int(nt/2)
    nc, nr = 3, 3
    plt.rcParams.update({'font.size': 12,'font.family':"sans-serif",\
                         'font.sans-serif':"Arial",'mathtext.default':"regular"})
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(10,12), sharex=True, sharey=True, subplot_kw=dict(projection=submap), 
                             layout = 'constrained')
    plt.suptitle('Image time = '+dt.datetime.strftime(submap.date.datetime, "%Y-%m-%dT%H:%M:%S"))
    fig.supxlabel('Solar X (arcsec)')
    fig.supylabel('Solar Y (arcsec)')
    cmap = plt.cm.get_cmap('cubehelix_r')

    for i, axi in enumerate(axes.flat):
        new_dem=(dem[:,:,i]+dem[:,:,i+1])/2.
        plotmap = Map(new_dem, submap.meta)
        plotmap.plot(axes=axi,norm=colors.LogNorm(vmin=1e19,vmax=1e22),cmap=cmap)
    
        y = axi.coords[1]
        y.set_axislabel(' ')
        if i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8:
            y.set_ticklabel_visible(False)
        x = axi.coords[0]
        x.set_axislabel(' ')
        if i < 6:
            x.set_ticklabel_visible(False)

        axi.set_title('Log(T) = {0:.2f} - {1:.2f}'.format(logtemps[i],logtemps[i+1]))

    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.98])
    plt.colorbar(ax=axes.ravel().tolist(),label='$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$',fraction=0.03, pad=0.02)
    plt.savefig(img_arr_tit, bbox_inches='tight')
    plt.close(fig)
    return



 
if __name__ == '__main__':
    set_start_method("spawn")
    ## Define constants and create the data directories
    data_disk = '/disk/solar/nn2/data/'
    data_disk_date = ['2018/10/28', '2018/10/29', '2018/10/30', '2018/10/31', '2018/11/01', '2018/11/02', '2018/11/03', '2018/11/04']
    #select only last 2 days
    # data_disk_date = data_disk_date[6:]
    output_dir = '/disk/solarz3/nn2/results/DEM_7day/'
    os.makedirs(output_dir, exist_ok='True')

    passband = [94, 131, 171, 193, 211, 335]

    for date in data_disk_date:
        print('Start Processing on date ' + date)
        start_time,end_time,ref_time,cadence,crd_cent,crd_width,ref_file_date,img_file_date,img_time_range = event_info(data_disk, date)

        hstart = img_time_range[0].hour
        hend = img_time_range[1].hour
        hour = np.arange(hstart,hend+1,1)
        start_hour = img_time_range[0]

        for h in hour:
            print('Processing data for hour ' + str(h))
            if h == hend:
                hr_time_range = [start_hour, start_hour+dt.timedelta(minutes=59)]
            else:
                hr_time_range = [start_hour, start_hour+dt.timedelta(hours=1)]
            # Check cadence for AIA
            for pband in passband:
                files, file_time = get_filelist_AIA(data_disk, pband, img_file_date, hr_time_range)
                n_img = ((hr_time_range[1]-hr_time_range[0]).total_seconds()/(cadence/u.second))

                if n_img > len(files):
                    print('Fewer than expected FITS files for '+str(pband)+' passband, Downloading data for '+str(pband)+' passband')
                    print('Download data for hour '+str(h))
                    get_data_AIA(start_hour, start_hour+dt.timedelta(hours=1), cadence, pband, data_disk+img_file_date, h, overwrite=False)
                else:
                    print('Data already downloaded for '+str(pband)+' passband')

            # print('Getting list of files')
            f_0094, time_0094 = get_filelist_AIA(data_disk, 94, img_file_date, hr_time_range)
            f_0131, time_0131 = get_filelist_AIA(data_disk, 131, img_file_date, hr_time_range)
            f_0171, time_0171 = get_filelist_AIA(data_disk, 171, img_file_date, hr_time_range)
            f_0193, time_0193 = get_filelist_AIA(data_disk, 193, img_file_date, hr_time_range)
            f_0211, time_0211 = get_filelist_AIA(data_disk, 211, img_file_date, hr_time_range)
            f_0335, time_0335 = get_filelist_AIA(data_disk, 335, img_file_date, hr_time_range)

            flist_pre = [f_0094, f_0131, f_0171, f_0193, f_0211, f_0335]
            time_pre = [time_0094, time_0131, time_0171, time_0193, time_0211, time_0335]

            # Reduce no. of file if there is more files than expected:
            for i in range(0, len(flist_pre)):
                if len(flist_pre[i]) > n_img:
                    flist_pre[i] = flist_pre[i][::50]
                    time_pre[i] = time_pre[i][::50]

            flength = [len(f_0094), len(f_0131), len(f_0171), len(f_0193), len(f_0211), len(f_0335)]
            flist = [f_0094, f_0131, f_0171, f_0193, f_0211, f_0335]
            time_array = [time_0094, time_0131, time_0171, time_0193, time_0211, time_0335]
            if max(flength) > n_img+1:
                print('More files than expected with number of images = '+str(max(flength))+', please check')
                break
            index = np.argmin(flength)
            # print(index)

            # Begin image processing
            start_img = closest(np.array(time_array[index][:]), start_hour)
            for img in range(start_img, len(flist[index])):

                print('Processing image, time = '+dt.datetime.strftime(time_array[index][img], "%Y-%m-%dT%H:%M:%S"))

            # Get and process images.
                err_arr_tit = output_dir+'error_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.asdf'
                map_arr_tit = output_dir+'prepped_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'_{index:03}.fits'
                # files = os.path.exists(err_arr_tit)
            #    if files == False:
                try:
                    map_array, err_array = prep_images(time_array,index,img,f_0094,f_0131,f_0171,f_0193,f_0211,f_0335,crd_cent,crd_width)
                except OSError as e:
                    print('{}'.format(e))
                    continue
            #       map_array.save(map_arr_tit,overwrite='True')
                #   tree = {'err_array':err_array}
                #   with asdf.AsdfFile(tree) as asdf_file:
                #     asdf_file.write_to(err_arr_tit, all_array_compression='zlib')
                
            #    else:
            #        print('Loading previously prepped images')
            #        arrs = asdf.open(err_arr_tit)
            #        err_array = arrs['err_array']
            #        ffin=sorted(glob.glob(output_dir+'prepped_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'*.fits'))
            #        map_array = Map(ffin)
                # Calculate DEMs
                dem_arr_tit = output_dir+'dem_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.asdf'
                files = os.path.exists(dem_arr_tit)
                if files == False:
                    print('Calculating DEM')
                    try:
                        dem,edem,elogt,chisq,dn_reg,mlogt,logtemps = calculate_dem(map_array,err_array)
                    except OSError as e:
                        print('{}'.format(e))
                        continue
                    print('DEM calculation completed, Saving asdf file')
                    tree = {'dem':dem, 'edem':edem, 'mlogt':mlogt, 'elogt':elogt, 'chisq':chisq, 'logtemps':logtemps}
                    # print('Tree defined')
                    with asdf.AsdfFile(tree) as asdf_file:  
                        asdf_file.write_to(dem_arr_tit, all_array_compression='zlib')
                    print('asdf file save as ' + dem_arr_tit)
                else:
                    print('Loading previously calculated DEM')
                    arrs = asdf.open(dem_arr_tit)  
                    dem = arrs['dem']
                    edem = arrs['edem']
                    mlogt = arrs['mlogt']
                    elogt = arrs['elogt']
                    chisq = arrs['chisq']
                    logtemps = arrs['logtemps']
                
                # Get a submap to have the scales and image properties.
                submap = get_submap(time_array,index,img,f_0193,crd_cent,crd_width)
                img_arr_tit = output_dir+'/DEM_image/DEM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
                plot = plot_dem_images(submap,dem,logtemps,img_arr_tit)
                print('DEM plotted')
                del dem, edem, mlogt, elogt, chisq, logtemps, map_array, err_array, submap
                print('delete variables, moving to next time step')
            
            print('Moving on to next hour')
            start_hour = start_hour+dt.timedelta(hours=1)
        
        print('Moving to next date')
    
    print('Job Done!')