# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Routine to get DEM from AIA data using Hannah & Kontar implementation

import os.path
import platform
import pdb
import datetime as dt
from aiapy.calibrate.prep import correct_degradation
import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
import astropy.time as time
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
import sunpy.map
from sunpy.net import Fido, attrs as a
import scipy.io as io
from dn2dem_pos import dn2dem_pos
import warnings
warnings.simplefilter('ignore')
from aiapy.calibrate import register, update_pointing, estimate_error, normalize_exposure
import aiapy.psf
import asdf

## Function with event information
def event_info():
    date = '2018/10/31'
    start_time = '00:00:00'
    end_time = '00:00:15' #'06:20:00'
    # bottom_left =  #[0, 0]*u.pixel 
    # top_right = [2450, 600]*u.pixel #[4095, 4095]*u.pixel 
    
    return date,start_time,end_time

## Function to download data
def get_data(date, start_time, end_time, passband, data_disk):
    ## Identify and download the data
    attrs_time = a.Time(date+' '+start_time, date+' '+end_time)
    for pbd in passband:
        wvlnth = a.Wavelength(int(pbd)*u.Angstrom, int(pbd)*u.Angstrom)
        result = Fido.search(attrs_time, a.Instrument('AIA'), wvlnth)
        files = Fido.fetch(result, path = data_disk+str(pbd).rjust(4, "0")+'/'+date, overwrite='False', progress='False')
    return
 
## Function to get the list of AIA filenames
def get_filelist(data_disk, passband):
    files = glob.glob(data_disk+str(passband).rjust(4, "0")+'/'+date+'/aia*.fits')
    files.sort()
    files_dt = [dt.datetime.strptime(file_i.split(f'{passband}a_')[1].split('z')[0], '%Y_%m_%dt%H_%M_%S_%f') for file_i in files]
    return files, files_dt

## Function to get closest file to defined file time.
def closest(list, current_time):
    ind = np.abs([time - current_time for time in list])
    return ind.argmin(0)

## Function to get AIA submap
def get_submap(time_array,index,img,f_0171):
    ind_0171 = closest(np.array(time_array[2][:]), time_array[index][img])
    aiamap = sunpy.map.Map(f_0171[ind_0171])
    bottom_left = SkyCoord(-650 * u.arcsec, -600 * u.arcsec, frame= aiamap.coordinate_frame)
    w1 = 800*u.arcsec
    h1 = 1200*u.arcsec
    submap = aiamap.submap(bottom_left, width=w1, height =h1)
    return submap

## Function to prep AIA images, deconvolve with PSF and produce submap
def prep_images(time_array,index,img,f_0094,f_0131,f_0171,f_0193,f_0211,f_0335):
    ind_0094 = closest(np.array(time_array[0][:]), time_array[index][img])
    ind_0131 = closest(np.array(time_array[1][:]), time_array[index][img])
    ind_0171 = closest(np.array(time_array[2][:]), time_array[index][img])
    ind_0193 = closest(np.array(time_array[3][:]), time_array[index][img])
    ind_0211 = closest(np.array(time_array[4][:]), time_array[index][img])
    ind_0335 = closest(np.array(time_array[5][:]), time_array[index][img])

    farray = [f_0094[ind_0094], f_0131[ind_0131], f_0171[ind_0171], f_0193[ind_0193], f_0211[ind_0211], f_0335[ind_0335]]
    maps = sunpy.map.Map(farray)
    bottom_left = SkyCoord(-650 * u.arcsec, -600 * u.arcsec, frame= maps[0].coordinate_frame)
    bottom_left_pix = skycoord_to_pixel(bottom_left, maps[0].wcs, origin = 0)
    w1 = 1333*u.pix
    h1 = 2000*u.pix
    sub = maps[0].submap(bottom_left_pix * u.pix, width=w1, height =h1)
    nx,ny = sub.data.shape
    nf=len(maps)

    print('Prepping images & deconvolving with PSF')
    map_arr = []
    error_array = np.zeros([nx, ny, nf])

    for m in range(0, len(maps)):
        psf = aiapy.psf.psf(maps[m].wavelength)
        aia_map_deconvolved = aiapy.psf.deconvolve(maps[m], psf=psf)
        aia_map_updated_pointing = update_pointing(aia_map_deconvolved)
        aia_map_registered = register(aia_map_updated_pointing)
        aia_map_corrected = correct_degradation(aia_map_registered)
        aia_map_norm = normalize_exposure(aia_map_corrected)
        bottom_left = SkyCoord(-650 * u.arcsec, -600 * u.arcsec, frame= maps[m].coordinate_frame)
        bottom_left_pix = skycoord_to_pixel(bottom_left, maps[m].wcs, origin = 0)
        submap = aia_map_norm.submap(bottom_left_pix * u.pix, width=w1, height =h1)
        map_arr.append(submap)
        num_pix=submap.data.size
        error_array[:,:,m] = estimate_error(submap.data*(u.ct/u.pix),submap.wavelength,num_pix)

    map_array=sunpy.map.Map(map_arr[0],map_arr[1],map_arr[2],map_arr[3],
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

    trin=io.readsav('/Users/nawinngampoopun/Desktop/Script/demreg/python/aia_tresp_en.dat')
    tresp_logt=np.array(trin['logt'])
    nt=len(tresp_logt)
    nf=len(trin['tr'][:])
    trmatrix=np.zeros((nt,nf))
    for i in range(0,nf):
        trmatrix[:,i]=trin['tr'][i]    
    
    t_space=0.05
    #probably should change to logT = 5.3-6.5 (Heinemann2021)
    t_min=5.3
    t_max=6.5
    logtemps=np.linspace(t_min,t_max,num=int((t_max-t_min)/t_space)+1)
    temps=10**logtemps
    mlogt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) for i in np.arange(0,len(temps)-1)])
    dem,edem,elogt,chisq,dn_reg=dn2dem_pos(image_array,err_array,trmatrix,tresp_logt,temps,max_iter=15)
    dem = dem.clip(min=0)
    return dem,edem,elogt,chisq,dn_reg,mlogt,logtemps

## Function to plot the DEM curve for the centre pixel
def plot_dem(dem,edem,mlogt,elogt,img_tit):
    sz = dem.shape
    pixel_loc = [int(sz[1]/2)-100, int(sz[0]/2)-100]
    fig = plt.figure(figsize=(8, 4.5))
    dem_pix = dem[pixel_loc[0],pixel_loc[1],]
    dem_pix_err = edem[pixel_loc[0],pixel_loc[1],]
    elogt_pix = elogt[pixel_loc[0],pixel_loc[1],]
    plt.errorbar(mlogt,dem_pix,xerr=elogt_pix,yerr=dem_pix_err, fmt='or',ecolor='lightcoral',elinewidth=3,capsize=0,label='Def Self LWght')
    plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
    plt.ylabel('$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$')
    plt.ylim([1e18,4e20])
    plt.xlim([5.3,6.7])
    plt.rcParams.update({'font.size': 16})
    plt.yscale('log')
    plt.legend()
    plt.savefig(img_tit,bbox_inches='tight')
    plt.close(fig)
    return

## Function to plot the DEM images
def plot_dem_images(submap,dem,logtemps,img_arr_tit):
    sz = dem.shape
    pixel_loc = [int(sz[1]/2)-100, int(sz[0]/2)-100]
    nt=len(dem[0,0,:])
    nt_new=int(nt/2)
    nc, nr = 3, 2
    plt.rcParams.update({'font.size': 12,'font.family':"sans-serif",\
                         'font.sans-serif':"Arial",'mathtext.default':"regular"})
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(12,6),sharex=True,sharey=True)
    fig.suptitle('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
    fig.supxlabel('Pixels')
    fig.supylabel('Pixels')
    cmap = plt.cm.get_cmap('cubehelix_r')

    for i, axi in enumerate(axes.flat):
        new_dem=(dem[:,:,i*4]+dem[:,:,i*4+3])/2.
        im = axi.imshow(new_dem,vmin=1e18,vmax=1e20,origin='lower',cmap=cmap,aspect='auto')
        axi.plot(pixel_loc[0],pixel_loc[1],'x',markersize='40')
        axi.set_title('{0:.2f} - {1:.2f}'.format(logtemps[i*4],logtemps[i*4+3+1]))

    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist(),label='$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$',fraction=0.03, pad=0.02)
    plt.savefig(img_arr_tit, bbox_inches='tight')
    plt.close(fig)
    return

#Calculate and plot EM maps for 9 temperature bins
def plot_em_images_calc(submap,dem,logtemps, img_EM_tit):
    EM_total = np.sum(dem,axis=2)
    EM_bin = np.zeros((dem.shape[0], dem.shape[1], dem.shape[2]//4))
    for i in range (EM_bin.shape[2]):
        EM_bin[:,:,i] = dem[:,:,i*4] + dem[:,:,i*4+3]
        
    nc, nr = 3, 2
    plt.rcParams.update({'font.size': 12,'font.family':"sans-serif",\
                         'font.sans-serif':"Arial",'mathtext.default':"regular"})
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(12,6),sharex=True,sharey=True)
    fig.suptitle('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
    fig.supxlabel('Pixels')
    fig.supylabel('Pixels')
    cmap = plt.cm.get_cmap('cubehelix_r')
    for i, axi in enumerate(axes.flat):
        im = axi.imshow(EM_bin[:,:,i],vmin=1e19,vmax=1e21,origin='lower',cmap=cmap, aspect='equal')
        axi.set_title('{0:.2f} - {1:.2f}'.format(logtemps[i*4],logtemps[i*4+3+1]))
    
    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist(),label='$\mathrm{EM\;[cm^{-5}]}$',fraction=0.03, pad=0.02)
    plt.savefig(img_EM_tit, bbox_inches='tight')
    plt.close(fig)
    
    return EM_total

#plot EM weighted temperature and density map based by Saqri(2020)
def plot_temp_images(submap,dem, EM_total,logtemps, img_temp_tit):
    temps = 10**(logtemps)
    upfrac = np.zeros((EM_total.shape))
    for j in range (dem.shape[2]):
        upfrac_plus = dem[:,:,j]*temps[j]
        upfrac = upfrac+upfrac_plus

    T_weighted = upfrac/EM_total
    T_min = np.nanmin(T_weighted)
    T_weighted_plot = np.nan_to_num(T_weighted,nan=0)
    
    plt.rcParams.update({'font.size': 12,'font.family':"sans-serif",\
                         'font.sans-serif':"Arial",'mathtext.default':"regular"})
    cmap = plt.cm.get_cmap('afmhot')
    cmap.set_bad(color='green')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection=submap)
    ax.set_title('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    im = ax.imshow(T_weighted_plot,vmin = T_min, vmax = 3e6,origin='lower',cmap=cmap,aspect='equal')
    plt.tight_layout()
    plt.colorbar(im,label='$\mathrm{T\;[K]}$',fraction=0.03, pad=0.02)
    plt.savefig(img_temp_tit, bbox_inches='tight')
    plt.close(fig)
    return

def plot_dens_images(submap, EM_total, img_dens_tit):
    #Scale Height = 42 Mm (Saqri 2020)
    h_saq = 42*(10**8) #cm
    #Calc mean density
    density_mean_saq = np.sqrt(EM_total/h_saq)
    dens_min = np.min(density_mean_saq[np.nonzero(density_mean_saq)])
    dens_max = np.max(density_mean_saq)
    
    plt.rcParams.update({'font.size': 12,'font.family':"sans-serif",\
                         'font.sans-serif':"Arial",'mathtext.default':"regular"})
    cmap = plt.cm.get_cmap('bone')
    cmap.set_bad(color='green')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection=submap)
    ax.set_title('Image time = '+time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    im = ax.imshow(density_mean_saq,vmin = dens_min, vmax = 1e6,origin='lower',cmap=cmap,aspect='equal')
    plt.tight_layout()
    plt.colorbar(im,label=r'$\bar{n} \mathrm{[cm^{-5}]}$',fraction=0.03, pad=0.02)
    plt.savefig(img_dens_tit, bbox_inches='tight')
    plt.close(fig)
    return



if __name__ == "__main__":
    ## Get the event information
    date,start_time,end_time = event_info()

    ## Define and create the data and output directories
    # if platform.system() == 'Darwin':
    #     data_disk = '/Users/davidlong/Data/SDO/'
    #     output_dir = '/Users/davidlong/idl_output/DEM/'+date+'/'
    # if platform.system() == 'Linux':
    #     data_disk = '/disk/solar15/dml/data/'
    #     output_dir = '/disk/corpita02/corpita/DEM/'+date+'/'

    data_disk = '/Users/nawinngampoopun/Desktop/SampleDEM/data/'
    output_dir = '/Users/nawinngampoopun/Desktop/SampleDEM/results/'

    os.makedirs(data_disk, exist_ok='True')
    os.makedirs(output_dir, exist_ok='True')
    passband = [94, 131, 171, 193, 211, 335]

    ## Download the data
    files = glob.glob(data_disk+str(passband[0]).rjust(4, "0")+'/'+date+'/*.fits')
    if len(files) == 0:
        print('Downloading data')
        get_data(date, start_time, end_time, passband, data_disk)
    else:
        print('Data already downloaded')

    ## Get list of files from each passband to identify the smallest number of files
    print('Getting list of files')
    f_0094, time_0094 = get_filelist(data_disk, 94)
    f_0131, time_0131 = get_filelist(data_disk, 131)
    f_0171, time_0171 = get_filelist(data_disk, 171)
    f_0193, time_0193 = get_filelist(data_disk, 193)
    f_0211, time_0211 = get_filelist(data_disk, 211)
    f_0335, time_0335 = get_filelist(data_disk, 335)

    flength = [len(f_0094), len(f_0131), len(f_0171), len(f_0193), len(f_0211), len(f_0335)]
    flist = [f_0094, f_0131, f_0171, f_0193, f_0211, f_0335]
    time_array = [time_0094, time_0131, time_0171, time_0193, time_0211, time_0335]
    index = np.argmin(flength)

    # Begin image processing
    for img in range(0, len(flist[index])):

        print('Processing image, time = '+dt.datetime.strftime(time_array[index][img], "%Y-%m-%dT%H:%M:%S"))

        # Get and process images.
        err_arr_tit = output_dir+'error_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.asdf'
        map_arr_tit = output_dir+'prepped_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'_{index:03}.fits'
        files = os.path.exists(err_arr_tit)
        if files == False:
            map_array, err_array = prep_images(time_array,index,img,f_0094,f_0131,f_0171,
                                               f_0193,f_0211,f_0335)
            map_array.save(map_arr_tit,overwrite='True')
            tree = {'err_array':err_array}
            with asdf.AsdfFile(tree) as asdf_file:
                asdf_file.write_to(err_arr_tit)
        else:
            print('Loading previously prepped images')
            arrs = asdf.open(err_arr_tit)
            err_array = arrs['err_array']
            ffin=sorted(glob.glob(output_dir+'prepped_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'*.fits'))
            map_array=sunpy.map.Map(ffin)
        # Calculate DEMs
        dem_arr_tit = output_dir+'dem_data_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.asdf'
        files = os.path.exists(dem_arr_tit)
        if files == False:
            print('Calculating DEM')
            dem,edem,elogt,chisq,dn_reg,mlogt,logtemps = calculate_dem(map_array,err_array)
            tree = {'dem':dem, 'edem':edem, 'mlogt':mlogt, 'elogt':elogt, 'chisq':chisq, 'logtemps':logtemps}
            with asdf.AsdfFile(tree) as asdf_file:  
                asdf_file.write_to(dem_arr_tit)
        else:
            print('Loading previously calculated DEM')
            arrs = asdf.open(dem_arr_tit)  
            dem = arrs['dem']
            edem = arrs['edem']
            mlogt = arrs['mlogt']
            elogt = arrs['elogt']
            chisq = arrs['chisq']
            logtemps = arrs['logtemps']

        # Plot results
        img_tit = output_dir+'Centre_pixel_DEM_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        plot_pix = plot_dem(dem,edem,mlogt,elogt,img_tit)

        # Get a submap to have the scales and image properties.
        submap = get_submap(time_array,index,img,f_0171)
        img_arr_tit = output_dir+'DEM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        plot_demmap = plot_dem_images(submap,dem,logtemps,img_arr_tit)
        img_EM_tit = output_dir+'EM_images_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        EM_total = plot_em_images_calc(submap,dem,logtemps,img_EM_tit)
        img_temp_tit = output_dir+'Temp_map_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        plot_tempmap = plot_temp_images(submap,dem,EM_total,logtemps,img_temp_tit)
        img_dens_tit = output_dir+'Dens_map_'+dt.datetime.strftime(time_array[index][img], "%Y%m%d_%H%M%S")+'.png'
        plot_densmap = plot_dens_images(submap, EM_total, img_dens_tit)

    #pdb.set_trace()
