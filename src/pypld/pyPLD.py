from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
import scipy.optimize as opt

import matplotlib.colors as colors
from skimage.measure import label, regionprops
import math

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})


def create_circular_mask(h, w, center=None, radius=None):
    #mask function used in limiting data
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



def read_elipsometry_data(filename, res, size_x, size_y,x_row=0,y_row=1,z_row=2,save=False, centre=(0,0),centre_line=False):
    
    #importing plume data
    data=np.loadtxt(filename,skiprows=1)
    X = data[:,x_row]+centre[0]
    Y = data[:,y_row]+centre[1]
    Z = data[:,z_row]
    # define grid.
    Xi = np.linspace(size_x[0],size_x[1],res)
    Yi = np.linspace(size_y[0],size_y[1],res)
    # grid the data.
    Zi = griddata((X, Y), Z, (Xi[None,:], Yi[:,None]), method='cubic')

    # contour the gridded data, plotting dots at the randomly spaced data points.
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots(1,2)
    ax[0].contour(Xi,Yi,Zi,15,linewidths=0.5,colors='k')
    im1=ax[0].contourf(Xi,Yi,Zi,15)
    ax[0].set_aspect('equal') 

    im2=ax[1].pcolormesh(Xi,Yi,Zi)
    ax[1].set_aspect('equal') 

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax)

    # plot data points.
    ax[0].scatter(X,Y,marker='o',c='b',s=5)
    if centre_line==True:
        ax[0].axhline(y=0, color='b', linestyle='-')
        ax[0].axvline(x=0, color='b', linestyle='-')
        
    print(np.min(Z))
    print(np.max(Z))

    
    if save==True:
        fig.savefig(fname='Load_fig_'+filename+'.png')
    
    
    return Xi, Yi, Zi


def find_rotation_angle_and_centre_of_plume(Xi, Yi, Zi, treshold, print_image=True):
    #make binary image with treshold and find angle of rotation by calculation imge moemnts of 
    binary_image=np.zeros(Zi.shape)
    binary_image[Zi>treshold]=1
    #lable image and find region props
    label_img = label(binary_image)
    regions = regionprops(label_img)
    
    rotation_angle = 90-regions[0].orientation*180/np.pi
    print('Image rotation needed:', rotation_angle)
    
    Shift_plume=[Xi[int(regions[0].centroid[0])],Yi[int(regions[0].centroid[1])]]
    print('Plume shift / centre:', Shift_plume)
    
    if print_image==True:
        fig, ax = plt.subplots(ncols=2, nrows=1,figsize=[14,7])
        ax[0].imshow(Zi)

        ax[0].contour(binary_image,1,colors='k')

        y0, x0 = regions[0].centroid
        orientation = regions[0].orientation
        x1 = x0 + math.cos(orientation-np.pi/2) * 0.5 * regions[0].minor_axis_length
        y1 = y0 - math.sin(orientation-np.pi/2) * 0.5 * regions[0].minor_axis_length
        x2 = x0 - math.sin(orientation-np.pi/2) * 0.5 * regions[0].major_axis_length
        y2 = y0 - math.cos(orientation-np.pi/2) * 0.5 * regions[0].major_axis_length

        ax[0].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax[0].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax[0].plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = regions[0].bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax[0].plot(bx, by, '-b', linewidth=2.5)
    
    #rotate binary image to find centre after roatiion
    binary_image_rot=ndimage.rotate(binary_image, (rotation_angle), reshape=False)
    #lable image and find region props
    label_img = label(binary_image_rot)
    regions = regionprops(label_img)

    #find centre/shift of plume after rotation
    Shift_after_rot=[Xi[int(regions[0].centroid[0])],Yi[int(regions[0].centroid[1])]]
    print('Plume shift / centre after rot:', Shift_after_rot)
    
    if print_image==True:
        Zi[np.isnan(Zi)]=0
        Zi_rot=ndimage.rotate(Zi, (rotation_angle), reshape=False)
        ax[1].contour(binary_image_rot,[0.5],colors='k')
        ax[1].imshow(Zi_rot)

        y0, x0 = regions[0].centroid
        orientation = regions[0].orientation
        x1 = x0 + math.cos(orientation-np.pi/2) * 0.5 * regions[0].minor_axis_length
        y1 = y0 - math.sin(orientation-np.pi/2) * 0.5 * regions[0].minor_axis_length
        x2 = x0 - math.sin(orientation-np.pi/2) * 0.5 * regions[0].major_axis_length
        y2 = y0 - math.cos(orientation-np.pi/2) * 0.5 * regions[0].major_axis_length

        ax[1].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax[1].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax[1].plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = regions[0].bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax[1].plot(bx, by, '-b', linewidth=2.5)

        plt.show()
    return Shift_after_rot,Shift_plume, rotation_angle


def curve_fit_plume(Xi,Yi,Zi, res_experimental, rotation_angle, n_pulses_plume, initial_guess, bounds , double=False):
    
    #function for curve fitting
    if double==False:
        def Plume(xy_mesh, T_0, z, n, m, shift_1, shift_2):

            # unpack 1D list into 2D x and y coords
            (xx, yy) = xy_mesh

            # make the 2D plume matrix
            plume= T_0*(z/np.sqrt(z**2+(xx-shift_1)**2))**n*(z/np.sqrt(z**2+(yy-shift_2)**2))**m

            radius = res_experimental/2*0.95 
            mask = create_circular_mask(res_experimental, res_experimental,radius=radius)
            plume[~mask] = 0

            # flatten the 2D plume down to 1D
            return np.ravel(plume)
    elif double==True:
        def Plume(xy_mesh, T_0, z, n_1, m_1,n_2,m_2, shift_1, shift_2):

            # unpack 1D list into 2D x and y coords
            (xx, yy) = xy_mesh

            # make the 2D plume matrix
            plume_1= T_0/2*(z/np.sqrt(z**2+(xx-shift_1)**2))**n_1*(z/np.sqrt(z**2+(yy-shift_2)**2))**m_1

            plume_2= T_0/2*(z/np.sqrt(z**2+(xx-shift_1)**2))**n_2*(z/np.sqrt(z**2+(yy-shift_2)**2))**m_2

            plume=plume_1+plume_2

            radius = res_experimental/2*0.95 
            mask = create_circular_mask(res_experimental, res_experimental,radius=radius)
            plume[~mask] = 0

            # flatten the 2D plume down to 1D
            return np.ravel(plume)

    
    xy_mesh = np.meshgrid(Xi,Yi)

    #import data, remove nans, negative values and mask data to avoid edge effects.
    fit_data=Zi
    fit_data[np.isnan(fit_data)]=0
    fit_data=ndimage.rotate(fit_data, rotation_angle, reshape=False)
    radius = res_experimental/2*0.95 #smaller res than wafer to avoid edge effects
    mask = create_circular_mask(res_experimental, res_experimental,radius=radius)
    fit_data[~mask] = 0
    fit_data[fit_data<0]=0

    #Fitting
    popt, pcov = opt.curve_fit(Plume, xy_mesh, np.ravel(fit_data), p0=initial_guess,bounds=bounds)
    
    #Plot results
    data_fitted =Plume(xy_mesh, *popt)
    data_fitted = data_fitted.reshape(res_experimental, res_experimental)
    
    popt[0]=popt[0]/n_pulses_plume
    
    return fit_data, data_fitted, popt


def curve_fit_plume_old(Xi,Yi,Zi, res_experimental, rotation_angle, n_pulses_plume, initial_guess, bounds ):
    
    #function for curve fitting
    def Plume(xy_mesh, T_0, z, n, m, shift_1, shift_2):

        # unpack 1D list into 2D x and y coords
        (xx, yy) = xy_mesh

        # make the 2D plume matrix
        plume= T_0*(z/np.sqrt(z**2+(xx-shift_1)**2))**n*(z/np.sqrt(z**2+(yy-shift_2)**2))**m

        radius = res_experimental/2*0.95 
        mask = create_circular_mask(res_experimental, res_experimental,radius=radius)
        plume[~mask] = 0

        # flatten the 2D plume down to 1D
        return np.ravel(plume)
    
    
    xy_mesh = np.meshgrid(Xi,Yi)

    #import data, remove nans, negative values and mask data to avoid edge effects.
    fit_data=Zi
    fit_data[np.isnan(fit_data)]=0
    fit_data=ndimage.rotate(fit_data, rotation_angle, reshape=False)
    radius = res_experimental/2*0.95 #smaller res than wafer to avoid edge effects
    mask = create_circular_mask(res_experimental, res_experimental,radius=radius)
    fit_data[~mask] = 0
    fit_data[fit_data<0]=0

    #Fitting
    popt, pcov = opt.curve_fit(Plume, xy_mesh, np.ravel(fit_data), p0=initial_guess,bounds=bounds)
    
    #Plot results
    data_fitted =Plume(xy_mesh, *popt)
    data_fitted = data_fitted.reshape(res_experimental, res_experimental)
    
    popt[0]=popt[0]/n_pulses_plume
    
    return fit_data, data_fitted, popt




def find_number_of_pulses(ang_step, print_output=False):

    #The aproximate number of pulses per substrate rotation
    aprox_pulses_per_rotation=int(360/ang_step)
    #degrees shift after aprox_pulses_per_rotation pulses
    rot_shift=round(360-ang_step*aprox_pulses_per_rotation,3)
    if abs(rot_shift)<10**(-4):
        substrate_rotations=1
        extra_pulses=0
        minimum_puls_num=aprox_pulses_per_rotation
        print('No shift in pulse placement. Concider changing angular steps between pulses. ')
    else:
        # find the number of rotations needed for the shift to equal a whole number of the angular step per pulse.  
        for i in range(1,10000):
            if abs(i*ang_step/rot_shift-round(i*ang_step/rot_shift,0)) < 10**(-10):
                substrate_rotations=int(round(i*ang_step/rot_shift,2))
                extra_pulses=i
                break
        # calculate the number of pulses needed to get the desired number of rounds. Add extra pulses to compesnate the shift:
        minimum_puls_num=substrate_rotations*aprox_pulses_per_rotation+extra_pulses
    if print_output==True:    
        print('The number of pulses per substrate rotation are: ',360/ang_step)
        print('Shift in degrees between substrate rotations is ',-rot_shift)
        print('Number of full substrate rotations needed to be be a whole number of pulses: ', substrate_rotations)
        print('Extra pulses neeeded: ', extra_pulses)
        print( minimum_puls_num*ang_step/360, 'whole rounds are completed in', minimum_puls_num, 'pulses.')

    return minimum_puls_num, extra_pulses, substrate_rotations,rot_shift, aprox_pulses_per_rotation



#define easy function to make plume based on fitted parameter
def def_plume(T_0, z, n, m, shift, dim, res, double=False):
    
    xx, yy = np.mgrid[-dim/2:dim/2:dim/res, -dim/2:dim/2:dim/res]
    if double==False:
        plume= T_0*(z/np.sqrt(z**2+(xx-shift[0])**2))**n*(z/np.sqrt(z**2+(yy-shift[1])**2))**m

    elif double==True:
        plume_1= T_0/2*(z/np.sqrt(z**2+(xx-shift[0])**2))**n[0]*(z/np.sqrt(z**2+(yy-shift[1])**2))**m[0]
        plume_2= T_0/2*(z/np.sqrt(z**2+(xx-shift[0])**2))**n[1]*(z/np.sqrt(z**2+(yy-shift[1])**2))**m[1]
        plume=plume_1+plume_2

    return plume



#simulate pld by adding a plume for all pulses with a  rotation in between
def simulate_pld(plume,n_pulses_model,ang_step):
    img = np.zeros(plume.shape)
    for i in np.arange(n_pulses_model):
        img = ndimage.rotate(img+plume, ang_step, reshape=False)
    return img


def simulate_pld_faster_old(plume,n_pulses,ang_step, print_output=True):
    pulses=find_number_of_pulses(ang_step,print_output=print_output)
    
    #Simulate first round of pulses (only the integre first pulses are deposited. Shift between rounds is handled in the next step.)
    first_PLD_substrate_round=simulate_pld(plume,pulses[4],ang_step)
    
    #Simulate deposition rounds needed to get a full integre number of pulses to coencide with a full number of substrate rotations. Add additional pulses to get a full number of substrate rotations. The rotation step is added to acount for the rotation of the additional pulses. 
    minimum_number_of_rounds=ndimage.rotate(simulate_pld(first_PLD_substrate_round,pulses[2],-pulses[3])+simulate_pld(plume,pulses[1],-ang_step), ang_step*pulses[1], reshape=False)
    
    #find the number of leftover pulses (after an integre number of the dimulated rounds above) to get to the total number of pulses
    number_leftover_pulses=int(pulses[0]*(n_pulses/pulses[0]-int(n_pulses/pulses[0])))
    
    if print_output==True:    
        print('Pulses in adittion to evenly distributed pulse number: ',number_leftover_pulses) 
        print('Suggested total number of pulses: ',n_pulses-number_leftover_pulses)
    #simulate the leftover pulses as a sum of first_pld_substrate_rounds and individual pulses. Here rotation of the last pulses is added to compesnate for rotation shift from the first_pld_substrate_rounds
    leftover_pulses=simulate_pld(first_PLD_substrate_round,int(number_leftover_pulses/pulses[4]),-pulses[3])+ndimage.rotate(simulate_pld(plume,number_leftover_pulses-int(number_leftover_pulses/pulses[4])*pulses[4],ang_step), -pulses[3]*int(number_leftover_pulses/pulses[4]), reshape=False)

    thinfilm =minimum_number_of_rounds*int(n_pulses/pulses[0])+leftover_pulses
    return thinfilm

def simulate_pld_faster(plume,n_pulses,ang_step, print_output=True):
    pulses=find_number_of_pulses(ang_step,print_output=print_output)
    
    #Simulate first round of pulses (only the integre first pulses are deposited. Shift between rounds is handled in the next step.)
    first_PLD_substrate_round=simulate_pld(plume,pulses[4],ang_step)
    
    #Simulate deposition rounds needed to get a full integre number of pulses to coencide with a full number of substrate rotations. Add additional pulses to get a full number of substrate rotations. ere rotation of the last pulses is added to compesnate for rotation shift from the first_pld_substrate_rounds
    rounds=simulate_pld(first_PLD_substrate_round,pulses[2],-pulses[3])
    extra_pulses=ndimage.rotate(simulate_pld(plume,pulses[1],ang_step), -(pulses[3])*(pulses[2]+1), reshape=False)
    minimum_number_of_rounds=rounds+extra_pulses
    
    #find the number of leftover pulses (after an integre number of the dimulated rounds above) to get to the total number of pulses
    number_leftover_pulses=int(round(pulses[0]*(n_pulses/pulses[0]-int(n_pulses/pulses[0])),0))
    
    if print_output==True:    
        print('Pulses in adittion to evenly distributed pulse number: ',number_leftover_pulses) 
        print('Suggested total number of pulses: ',n_pulses-number_leftover_pulses)
    #simulate the leftover pulses as a sum of first_pld_substrate_rounds and individual pulses. Here rotation of the last pulses is added to compesnate for rotation shift from the first_pld_substrate_rounds
    leftover_pulses=simulate_pld(first_PLD_substrate_round,int(number_leftover_pulses/pulses[4]),-pulses[3])+ndimage.rotate(simulate_pld(plume,number_leftover_pulses-int(number_leftover_pulses/pulses[4])*pulses[4],ang_step), -pulses[3]*int(number_leftover_pulses/pulses[4]), reshape=False)
        
    thinfilm =minimum_number_of_rounds*int(n_pulses/pulses[0])+leftover_pulses
    return thinfilm

def plot_cb(x,y,image):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.subplots(1,1)

    im1=ax.pcolormesh(x,y,image)
    ax.set_aspect('equal')
    fig.colorbar(im1)
    plt.show()


def plot_compare(Xi, Yi, data, model, res, data_lable='Data', model_lable='Model', error_lable='error',save=False,filename='', fig_for_print=False,mask_scaling=0.93, N_contours=10,D_ticks_cbar=50, D_tick_error_cbar=5):
    
    #error
    radius = res/2*mask_scaling # sligtly smaler dim than curve fit to avoid edge effects
    mask = create_circular_mask(h=res, w=res,radius=radius)
    data_for_error_calculation=data.copy()
    data_for_error_calculation[~mask] = 1
    error=100*(model-data)/data_for_error_calculation
    #error=model-data

    error[~mask] = np.nan
    data[~mask] = np.nan
    model[~mask] = np.nan
    
    
    if fig_for_print==True:
        fig = plt.figure(figsize=(7.5,2.5), dpi=600)    
        plt.rcParams.update({'font.size': 9})
        plt.rcParams.update({'xtick.labelsize': 8})
        plt.rcParams.update({'ytick.labelsize': 8})
        
    else:
        fig = plt.figure(figsize=(28,7))    
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})
    
    ax = fig.subplots(1,4,sharey=True)
    
    data_min=min([np.nanmin(data),np.nanmin(model)])
    data_max=max([np.nanmax(data),np.nanmax(model)])
    ticks=np.round(np.arange(data_min,data_max,D_ticks_cbar)/D_ticks_cbar)*D_ticks_cbar
    
    #plot model
    im3=ax[1].pcolormesh(Xi,Yi,model,vmin=data_min,vmax=data_max)
    fig.colorbar(im3, ax=ax[1],orientation='horizontal',ticks=ticks, pad=0.2, label='[nm]')
    ax[1].set_title(model_lable)
    ax[1].set_aspect('equal')
    
    #plot data
    im2=ax[0].pcolormesh(Xi,Yi,data,vmin=data_min,vmax=data_max)
    fig.colorbar(im2, ax=ax[0],orientation='horizontal',ticks=ticks, pad=0.2, label='[nm]')
    ax[0].set_title(data_lable)
    ax[0].set_aspect('equal') 

    #plot contour comaprison
    contours=np.arange(np.nanmin(data),np.nanmax(data),(np.nanmax(data)-np.nanmin(data))/N_contours)
    
    ax[2].contour(Xi,Yi,model, contours, colors='r')
    ax[2].contour(Xi,Yi,data, contours,colors='w')
    
    im1=ax[2].pcolormesh(Xi,Yi,data,vmin=data_min,vmax=data_max)
    fig.colorbar(im2, ax=ax[2],orientation='horizontal',ticks=ticks, pad=0.2, label='[nm]')
#    ax[2].set_title(data_lable+' with counture plot of '+model_lable+' in red')
    ax[2].set_title('Contoure plot')
    ax[2].set_aspect('equal')
    
    error_min=np.nanmin(error)
    error_max=np.nanmax(error)
    ticks_error=np.round(np.arange(error_min,error_max,D_tick_error_cbar)/D_tick_error_cbar)*D_tick_error_cbar
    
    
    im4=ax[3].pcolormesh(Xi,Yi,error)
    fig.colorbar(im4, ax=ax[3],orientation='horizontal',ticks=ticks_error, pad=0.2, label='[%]')
    ax[3].set_title(error_lable)
    ax[3].set_aspect('equal')
    
    plt.setp(ax[:], xlabel='[cm]')
    plt.setp(ax[0], ylabel='[cm]')
    
    if save==True:
        fig.savefig(fname=filename+'.png')

    if fig_for_print==True:
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})
    
def plot_compare_new(Xi, Yi, data, model, res, data_lable='Data', model_lable='Model', error_lable='error',save=False,filename='', fig_for_print=False,mask_scaling=0.93, N_contours=10,D_ticks_cbar=50, D_tick_error_cbar=5):
    
    #error
    radius = res/2*mask_scaling # sligtly smaler dim than curve fit to avoid edge effects
    mask = create_circular_mask(h=res, w=res,radius=radius)
    data_for_error_calculation=data.copy()
    data_for_error_calculation[~mask] = 1
    error=100*(model-data)/data_for_error_calculation
    #error=model-data

    error[~mask] = np.nan
    data[~mask] = np.nan
    model[~mask] = np.nan
    
    
    if fig_for_print==True:
        fig = plt.figure(figsize=(7.5,3), dpi=600)    
        plt.rcParams.update({'font.size': 9})
        plt.rcParams.update({'xtick.labelsize': 8})
        plt.rcParams.update({'ytick.labelsize': 8})
        
    else:
        fig = plt.figure(figsize=(28,7))    
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})
    
    ax0 = plt.subplot2grid((7, 20), (0, 0), colspan=5, rowspan=5)
    ax1 = plt.subplot2grid((7, 20), (0, 5), colspan=5, rowspan=5)
    ax2 = plt.subplot2grid((7, 20), (0, 10), colspan=5, rowspan=5)
    ax3 = plt.subplot2grid((7, 20), (0, 15), colspan=5, rowspan=5)
    
    cbar_ax1 = plt.subplot2grid((7, 20), (6, 1), colspan=13, rowspan=1)    
    cbar_ax2 = plt.subplot2grid((7, 20), (6, 15), colspan=5, rowspan=1)    

    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    
    data_min=min([np.nanmin(data),np.nanmin(model)])
    data_max=max([np.nanmax(data),np.nanmax(model)])
    ticks=np.round(np.arange(data_min,data_max,D_ticks_cbar)/D_ticks_cbar)*D_ticks_cbar
    
    #plot model
    im3=ax1.pcolormesh(Xi,Yi,model,vmin=data_min,vmax=data_max)
    fig.colorbar(im3, cax=cbar_ax1,orientation='horizontal',ticks=ticks, pad=0.2, label='Thikness [nm]', aspect=40)

    ax1.set_title(model_lable)
    ax1.set_aspect('equal')
    
    #plot data
    im2=ax0.pcolormesh(Xi,Yi,data,vmin=data_min,vmax=data_max)
    #fig.colorbar(im2, ax=ax[0],orientation='horizontal',ticks=ticks, pad=0.2, label='[nm]')
    ax0.set_title(data_lable)
    ax0.set_aspect('equal') 

    #plot contour comaprison
    contours=np.arange(np.nanmin(data),np.nanmax(data),(np.nanmax(data)-np.nanmin(data))/N_contours)
    
    ax2.contour(Xi,Yi,model, contours, colors='r')
    ax2.contour(Xi,Yi,data, contours,colors='w')
    
    im1=ax2.pcolormesh(Xi,Yi,data,vmin=data_min,vmax=data_max)
#    fig.colorbar(im2, ax=ax[2],orientation='horizontal',ticks=ticks, pad=0.2, label='[nm]')
#    ax[2].set_title(data_lable+' with counture plot of '+model_lable+' in red')
    ax2.set_title('Contoure plot')
    ax2.set_aspect('equal')
    
    error_min=np.nanmin(error)
    error_max=np.nanmax(error)
    ticks_error=np.round(np.arange(error_min,error_max,D_tick_error_cbar)/D_tick_error_cbar)*D_tick_error_cbar
    
    
    im4=ax3.pcolormesh(Xi,Yi,error)
    fig.colorbar(im4, cax=cbar_ax2,orientation='horizontal',ticks=ticks_error, pad=0.2, label='Error [%]')
    ax3.set_title(error_lable)
    ax3.set_aspect('equal')
    
    plt.setp(ax0, xlabel='[cm]')
    plt.setp(ax1, xlabel='[cm]')
    plt.setp(ax2, xlabel='[cm]')
    plt.setp(ax3, xlabel='[cm]')

    plt.setp(ax0, ylabel='[cm]')
    plt.tight_layout(w_pad=0.005, h_pad=0.005)
    
    if save==True:
        fig.savefig(fname=filename+'.png')

    if fig_for_print==True:
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'xtick.labelsize': 12})
        plt.rcParams.update({'ytick.labelsize': 12})

#Not needed?:
def cut_out_2inch_wafer(xx,yy,thinfilm,res,dim,sample_holder=None):
    ##Cut out wafer
    h, w = thinfilm.shape[:2]
    radius = 2.5*res/dim #cm * pixesls/cm

    mask = create_circular_mask(h, w,radius=radius)
    masked_img = thinfilm.copy()
    masked_img[~mask] = np.nan


    #plot results
    fig = plt.figure(figsize=(7, 7))
    ax = fig.subplots(1,1)
    im1=ax.pcolormesh(xx,yy,masked_img)
    ax.set_aspect('equal')
    plt.colorbar(im1)
    
    ax.set_xlim(-radius*dim/res,radius*dim/res)
    ax.set_ylim(-radius*dim/res,radius*dim/res)
    

    if sample_holder is not None:
        for s in sample_holder:
            ax.add_patch(matplotlib.patches.Rectangle(((s[0]-res/2)*dim/res,(s[2]-res/2)*dim/res), (s[1]-res/2)*dim/res-(s[0]-res/2)*dim/res, (s[3]-res/2)*dim/res-(s[2]-res/2)*dim/res,fill=False))
    return masked_img


def Plot_array_of_thikness_dif(xx,yy,thinfilms,shift_X, shift_Y, sample_centre, sample_edge, filename='',save=False):
    
    thicness_dif_edges =np.empty([thinfilms.shape[0],thinfilms.shape[1]])
    thicness_dif_centre =np.empty([thinfilms.shape[0],thinfilms.shape[1]])

    vmin_centre =np.empty([thinfilms.shape[0],thinfilms.shape[1]])
    vmax_centre =np.empty([thinfilms.shape[0],thinfilms.shape[1]])

    vmax_edges =np.empty([thinfilms.shape[0],thinfilms.shape[1]])
    vmin_edges =np.empty([thinfilms.shape[0],thinfilms.shape[1]])


    for i  in np.arange(thinfilms.shape[0]):
        for j  in np.arange(thinfilms.shape[1]):
            film=thinfilms[i,j,:,:]
            vmin_edges[i,j]=np.amin(film[sample_edge[0]:sample_edge[1],sample_edge[2]:sample_edge[3]])
            vmax_edges[i,j]=np.amax(film[sample_edge[0]:sample_edge[1],sample_edge[2]:sample_edge[3]])

            vmin_centre[i,j]=np.amin(film[sample_centre[0]:sample_centre[1],sample_centre[2]:sample_centre[3]])
            vmax_centre[i,j]=np.amax(film[sample_centre[0]:sample_centre[1],sample_centre[2]:sample_centre[3]])


            thicness_dif_edges[i,j]=(vmax_edges[i,j]-vmin_edges[i,j])/vmax_edges[i,j]
            thicness_dif_centre[i,j]=(vmax_centre[i,j]-vmin_centre[i,j])/vmax_centre[i,j]



    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'xtick.labelsize': 8})
    plt.rcParams.update({'ytick.labelsize': 8})


    fig = plt.figure(figsize=(45, 25))
    ax = fig.subplots(2,3)
    ax[0,0].matshow(vmin_centre, cmap=plt.cm.Blues)
    ax[0,0].set_title('vmin_centre')


    ax[0,1].matshow(vmax_centre, cmap=plt.cm.Blues)
    ax[0,1].set_title('vmax_centre')

    ax[0,2].matshow(thicness_dif_centre, cmap=plt.cm.Blues)
    ax[0,2].set_title('thickness_dif_centre [%]')




    ax[1,0].matshow(vmin_edges, cmap=plt.cm.Blues)
    ax[1,0].set_title('vmin_edges')

    ax[1,1].matshow(vmax_edges, cmap=plt.cm.Blues)
    ax[1,1].set_title('vmax_edges')

    ax[1,2].matshow(thicness_dif_edges, cmap=plt.cm.Blues)
    ax[1,2].set_title('thickness_dif_edges [%]')



    # higlight value with smallest dif for edge
    highlight_edge = np.zeros([thinfilms.shape[0],thinfilms.shape[1]])
    highlight_edge[np.argwhere(thicness_dif_edges==np.min(thicness_dif_edges))[0,0],np.argwhere(thicness_dif_edges==np.min(thicness_dif_edges))[0,1]]=1

    # higlight value with smallest dif for centre
    highlight_centre = np.zeros([thinfilms.shape[0],thinfilms.shape[1]])
    highlight_centre[np.argwhere(thicness_dif_centre==np.min(thicness_dif_centre))[0,0],np.argwhere(thicness_dif_centre==np.min(thicness_dif_centre))[0,1]]=1


    # RGBA overlay matrix
    overlay_edge = np.zeros([thinfilms.shape[0],thinfilms.shape[1],4])
    overlay_centre = np.zeros([thinfilms.shape[0],thinfilms.shape[1],4])

    # we set the red channel to 1
    overlay_edge[...,0] = 1.
    overlay_centre[...,0] = 1.

    # and we set the alpha to our boolean matrix 'highlight' so that it is
    # transparent except for highlighted pixels
    overlay_edge[...,3] = highlight_edge
    overlay_centre[...,3] = highlight_centre


    #loop over plots
    for i in np.arange(3):
        for j in np.arange(2):
            #set ticks
            ax[j,i].set_xticks(np.arange(shift_Y.shape[0]))
            ax[j,i].set_xticklabels(shift_Y.round(1))

            ax[j,i].set_yticks(np.arange(shift_X.shape[0]))
            ax[j,i].set_yticklabels(shift_X.round(1))

            ax[j,i].set_xlabel('y-shift')
            ax[j,i].set_ylabel('x-shift')



    #higlight
    for i in np.arange(3):
        ax[0,i].imshow(overlay_centre,interpolation='nearest')
        ax[1,i].imshow(overlay_edge,interpolation='nearest')


    #loop over arays to add values to plots
    for i in np.arange(thinfilms.shape[1]):
        for j in np.arange(thinfilms.shape[0]):
            c = vmin_centre[j,i]
            ax[0,0].text(i, j, '{0:.{1}f}'.format(c, 0), va='center', ha='center')
            c = vmax_centre[j,i]
            ax[0,1].text(i, j, '{0:.{1}f}'.format(c, 0), va='center', ha='center')

            c = vmin_edges[j,i]
            ax[1,0].text(i, j, '{0:.{1}f}'.format(c, 0), va='center', ha='center')
            c = vmax_edges[j,i]
            ax[1,1].text(i, j, '{0:.{1}f}'.format(c, 0), va='center', ha='center')

            c = thicness_dif_edges[j,i]*100
            ax[1,2].text(i, j, '{0:.{1}f}'.format(c, 1), va='center', ha='center')
            c = thicness_dif_centre[j,i]*100
            ax[0,2].text(i, j, '{0:.{1}f}'.format(c, 1), va='center', ha='center')


    if save==True:
        fig.savefig(fname='Thicness_comparisson'+filename+'.png')

        
def plot_array_of_films(xx,yy,thinfilms,res,dim,radius,filename='',save=False, sample_holder=None):

    # settings
    nrows, ncols = thinfilms.shape[0], thinfilms.shape[1]  # array of sub-plots
    figsize = [24, 24]     # figure size, inches

    thinfilms_concentrated=thinfilms.reshape([thinfilms.shape[0]*thinfilms.shape[1],thinfilms.shape[2],thinfilms.shape[3]])
    
    vmin=np.min(thinfilms)
    vmax=np.max(thinfilms)
    
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        im=axi.pcolormesh(xx,yy,thinfilms_concentrated[i], vmin=vmin, vmax=vmax)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

        
        if sample_holder is not None:
            for s in sample_holder:
                axi.add_patch(matplotlib.patches.Rectangle(((s[0]-res/2)*dim/res,(s[2]-res/2)*dim/res), (s[1]-res/2)*dim/res-(s[0]-res/2)*dim/res, (s[3]-res/2)*dim/res-(s[2]-res/2)*dim/res,fill=False))
        
        axi.set_xlim(-radius,radius)
        axi.set_ylim(-radius,radius)

    plt.tight_layout()

    #add colourbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


    if save==True:
        fig.savefig(fname='Film_comparisson'+filename+'.png')
        
        

def plot_countour_compare(xx,yy,thinfilm,Xi,Yi,data,dim_experimental, contours):
    ##Cut out wafer

    model=thinfilm*np.nanmax(data)/np.nanmax(thinfilm) #scaling factor for thikness

    fig = plt.figure(figsize=(24, 10))
    ax = fig.subplots(1,2)


    c1=ax[0].contour(Xi, Yi, data,contours,linewidths=0.5,colors='k')
    im1=ax[0].contourf(Xi, Yi, data,contours)

    ax[0].set_aspect('equal')
    ax[0].clabel(c1, c1.levels,fmt='%1.1f')    



    c2=ax[1].contour(xx,yy,model,contours,linewidths=0.5,colors='k')
    im2=ax[1].contourf(xx,yy,model,contours)
    
    ax[1].set_xlim(-dim_experimental/2,dim_experimental/2)
    ax[1].set_ylim(-dim_experimental/2,dim_experimental/2)
    ax[1].set_aspect('equal')
    ax[1].clabel(c2, c2.levels,fmt='%1.1f')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    
    
    
def plot_array_of_1d_plots_trough_centre_of_films(xx,yy,thinfilms,res,dim,radius,filename='',save=False):
    # settings
    nrows, ncols = thinfilms.shape[0], thinfilms.shape[1]  # array of sub-plots
    figsize = [24, 24]     # figure size, inches

    thinfilms_concentrated=thinfilms.reshape([thinfilms.shape[0]*thinfilms.shape[1],thinfilms.shape[2],thinfilms.shape[3]])

    vmin=np.min(thinfilms)
    vmax=np.max(thinfilms)
    
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # plot x and y  1D-plot  on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        im=axi.plot(yy[int(res/2),:],thinfilms_concentrated[i][int(res/2),:])
        im=axi.plot(yy[int(res/2),:],thinfilms_concentrated[i][:,int(res/2)])

        axi.set_ylim(vmin,vmax)
        axi.set_xlim(-radius,radius)

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))


    plt.tight_layout()
    

    plt.show()

    if save==True:
        fig.savefig(fname='1D_Film_comparisson'+filename+'.png')