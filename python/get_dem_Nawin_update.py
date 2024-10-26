# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    start_time = '2018/11/01 03:00:00'
    end_time = '2018/11/01 04:00:00'
    cadence = 10*u.second #seconds
    img_time_range = [dt.datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S"), dt.datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")]

    ref_time = '2018/11/01 03:00:04'
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
    bottom_left = SkyCoord(-400 * u.arcsec, -600 * u.arcsec, frame= map.coordinate_frame)
    bottom_left_pix = skycoord_to_pixel(bottom_left, map.wcs, origin = 0)*u.pixel
    top_right = SkyCoord(300 * u.arcsec, 600 * u.arcsec, frame= map.coordinate_frame)
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

# #Calculate and plot total EM
# def plot_em_images_calc(submap,dem,logtemps, img_EM_tit):
#     temps = 10**(logtemps)
#     EM_temp = np.zeros((dem.shape[0],dem.shape[1]))
#     for j in range ((dem.shape[2])-1):
#         #EM = sum(dem*deltaT)
#         EM_temp = EM_temp + dem[:,:,j]*(temps[j+1]-temps[j])
    
#     EM_total = EM_temp
#     EM_min = np.min(EM_total[np.nonzero(EM_total)])
#     #EM_max = np.max(EM_total)
        
#     plt.rcParams.update({'font.size': 12,'mathtext.default':"regular"})
#     cmap = plt.cm.get_cmap('plasma')
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection=submap)
#     ax.set_title('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
#     ax.set_xlabel('arcsec')
#     ax.set_ylabel('arcsec')
#     im = ax.imshow(EM_total,vmin = EM_min, vmax=1e27,origin='lower',cmap=cmap,aspect='equal')
#     # ax.contour(CHB, colors = 'black',linewidths=0.5)
#     # ax.contour(CHB_in,colors = 'red',linewidths =0.5)
#     # ax.contour(CHB_out, colors = 'blue',linewidths =0.5)
#     plt.tight_layout()
#     plt.colorbar(im,label='$\mathrm{EM\;[cm^{-5}]}$',fraction=0.03, pad=0.02)
#     plt.savefig(img_EM_tit, bbox_inches='tight')
#     plt.close(fig)
    
#     return EM_total

# #plot EM weighted temperature and density map based by Saqri(2020)
# def plot_temp_images(submap,dem, EM_total,logtemps, img_temp_tit):
#     temps = 10**(logtemps)
#     upfrac = np.zeros((EM_total.shape))
#     #T = (sum(DEM*deltaT)/EM)
#     for j in range (dem.shape[2]-1):
#         upfrac_plus = dem[:,:,j]*temps[j]*(temps[j+1]-temps[j])
#         upfrac = upfrac+upfrac_plus

#     T_weighted = upfrac/EM_total
#     T_min = np.nanmin(T_weighted)
#     T_weighted_plot = np.nan_to_num(T_weighted,nan=0)
    
#     plt.rcParams.update({'font.size': 12,'mathtext.default':"regular"})
#     cmap = plt.cm.get_cmap('gist_heat')
#     cmap.set_bad(color='green')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection=submap)
#     ax.set_title('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
#     ax.set_xlabel('arcsec')
#     ax.set_ylabel('arcsec')
#     im = ax.imshow(T_weighted_plot,vmin = T_min, vmax = 2e6,origin='lower',cmap=cmap,aspect='equal')
#     # ax.contour(CHB, colors = 'black',linewidths=0.5)
#     # ax.contour(CHB_in,colors = 'red',linewidths =0.5)
#     # ax.contour(CHB_out, colors = 'blue',linewidths =0.5)
#     plt.tight_layout()
#     plt.colorbar(im,label='$\mathrm{T\;[K]}$',fraction=0.03, pad=0.02)
#     plt.savefig(img_temp_tit, bbox_inches='tight')
#     plt.close(fig)
#     return T_weighted_plot

# def plot_dens_images(submap, EM_total, img_dens_tit):
#     #Scale Height h = 42 Mm (Saqri 2020)
#     h_saq = 42*(10**8) #cm
#     #Calc mean density n = sqrt(EM/h)
#     density_mean_saq = np.sqrt(EM_total/h_saq)
#     dens_min = np.min(density_mean_saq[np.nonzero(density_mean_saq)])
#     #dens_max = np.max(density_mean_saq)
    
#     plt.rcParams.update({'font.size': 12,'mathtext.default':"regular"})
#     cmap = plt.cm.get_cmap('bone')
#     cmap.set_bad(color='green')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection=submap)
#     ax.set_title('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
#     ax.set_xlabel('arcsec')
#     ax.set_ylabel('arcsec')
#     im = ax.imshow(density_mean_saq,vmin = dens_min, vmax = 3e8,origin='lower',cmap=cmap,aspect='equal')
#     # ax.contour(CHB, colors = 'black',linewidths=0.5)
#     # ax.contour(CHB_in,colors = 'red',linewidths =0.5)
#     # ax.contour(CHB_out, colors = 'blue',linewidths =0.5)
#     plt.tight_layout()
#     plt.colorbar(im,label=r'$\bar{n}\mathrm{\;[cm^{-3}]}$',fraction=0.03, pad=0.02)
#     plt.savefig(img_dens_tit, bbox_inches='tight')
#     plt.close(fig)
#     return density_mean_saq

 
if __name__ == '__main__':
    set_start_method("forkserver")
    ## Define constants and create the data directories
    data_disk = '/disk/solarz3/nn2/data_CH2018/2018/11/01/03/AIA'

    # os.makedirs(data_disk, exist_ok='True')

    ## Get the event information
    start_time,end_time,ref_time,cadence,crd_cent,crd_width,ref_file_date,img_file_date,img_time_range = event_info(data_disk)

    ## Define and create the output directories

    output_dir = '/disk/solarz3/nn2/results/DEM_highres/'

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
    # print(index)

    # Begin image processing
    start_img = closest(np.array(time_array[index][:]), dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S"))
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
        img_arr_tit = output_dir+'DEM_image/DEM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        plot = plot_dem_images(submap,dem,logtemps,img_arr_tit)
        # img_arr_tit = output_dir+'DEM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        # # plot_dem_images(submap,dem,logtemps,img_arr_tit,CHB, CHB_in, CHB_out)
        # plot_dem_images(submap,dem,logtemps,img_arr_tit)
        # img_EM_tit = output_dir+'EM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        # # EM_total = plot_em_images_calc(submap,dem,logtemps,img_EM_tit,CHB, CHB_in, CHB_out)
        # # EM_total = plot_em_images_calc(submap,dem,logtemps,img_EM_tit)
        # # Maybe using ASDF
        # tree = {'EM_total':EM_total}
        # with asdf.AsdfFile(tree) as asdf_file:
        #     asdf_file.write_to(output_dir+'EM_array_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.asdf', all_array_compression='zlib')
        # img_temp_tit = output_dir+'Temp_map_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        # # Tempmap = plot_temp_images(submap,dem,EM_total,logtemps,img_temp_tit,CHB, CHB_in, CHB_out)
        # Tempmap = plot_temp_images(submap,dem,EM_total,logtemps,img_temp_tit)
        # # np.savez(output_dir+'Temp_array_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.npz',Tempmap = Tempmap)
        # img_dens_tit = output_dir+'Dens_map_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        # # Densmap = plot_dens_images(submap, EM_total, img_dens_tit,CHB, CHB_in, CHB_out)
        # Densmap = plot_dens_images(submap, EM_total, img_dens_tit)
        print('DEM plotted')
        del dem, edem, mlogt, elogt, chisq, logtemps, map_array, err_array, submap
        print('delete variables, moving to next time step')
    print('Job Done!')