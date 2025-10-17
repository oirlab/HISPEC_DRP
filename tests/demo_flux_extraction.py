import numpy as np
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt

def _polynomial_smooth(data, deg=4):
    coeffs = np.zeros((data.shape[0],deg+1))
    smooth_data = np.zeros(data.shape)
    xpts = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        ypts = data[i]
        goodinds = np.isfinite(ypts)
        coeffs[i] = np.polyfit(xpts[goodinds], ypts[goodinds], deg)
        fit = np.poly1d(coeffs[i])
        smooth_data[i,:] = fit(xpts)
    return smooth_data, coeffs
    

def find_trace_locs(data):
    threshold_peak = 500
    cutsize = 10
    deg = 4

    ### Initial peak finding
    peaks,_ = signal.find_peaks(data[:,0],height=threshold_peak)
    
    ### Declare output arrays
    trace_locs = np.zeros((len(peaks),data.shape[1]))
    trace_widths = np.zeros(trace_locs.shape)
    smooth_trace_locs = np.zeros(trace_locs.shape)
    smooth_trace_widths = np.zeros(trace_locs.shape)
    # print(trace_locs.shape)
    # print(smooth_trace_locs.shape)
    coeffs = np.zeros((trace_locs.shape[0],deg+1))

    ### Loop through columns
    for i in range(data.shape[1]):
        ### Get index of peaks from scipy
        ### We could write our own jit-compatible version, but it's so
        ### fast that it doesn't seem worth it. The scipy version also
        ### has prominence/width criteria which might be helpful for
        ### avoiding bad pixels, once we have the actual assembled 
        ### detectors to check
        ### There are a number of additional checks we may need to add if 
        ### real fiber flats are noisy - the simulations are so photon-
        ### dominated that it's hard to be sure waht's needed
        peaks, _ = signal.find_peaks(data[:,i],height=threshold_peak)
        ### Get flux-weighted center for each peak on its own
        for j, peak in enumerate(peaks):
            inds = np.arange(peak-cutsize,peak+cutsize+1)
            dslice = data[inds,i]

            trace_locs[j,i] = np.sum(inds*dslice)/np.sum(dslice)
            trace_widths[j,i] = np.sqrt(np.sum((inds-peak)**2 * dslice)/np.sum(dslice))
            ### This is to reject bad pixels, will want to tune this value
            if trace_widths[j,i] > 1.9:
                trace_widths[j,i] = np.nan
            # print(trace_widths[j,i])
            # plt.axvline(-2.355/2*trace_widths[j,i])
            # plt.axvline(2.355/2*trace_widths[j,i])
            # plt.axhline(0.5*np.max(dslice))
            # plt.plot(inds-trace_locs[j,i],dslice)
            # plt.show()


    ### Do polynomial fit/smoothing to traces
    smooth_trace_locs, loc_coeffs = _polynomial_smooth(trace_locs,deg=deg)
    ### Convert to FWHM and smooth
    smooth_trace_widths, width_coeffs = _polynomial_smooth(2.355*trace_widths, deg=deg)

    return smooth_trace_locs, loc_coeffs, smooth_trace_widths, width_coeffs

def fit_trace_prof(data, trace_pos, width):
    ### Instantiate output array - norder x npix x width
    out = np.zeros((trace_pos.shape[0],trace_pos.shape[1], 2*width+1))
    ### Loop over orders
    for i in range(trace_pos.shape[0]):
        ### Loop over pixels
        for j in range(trace_pos.shape[1]):
            ### Slice out the data within +/- width of the center
            center = int(trace_pos[i,j])
            dslice = data[center-width:center+width+1, j]
            ### Normalize so that it sums to 1 and save
            out[i,j,:] = dslice/np.nansum(dslice)
    return out

def extract_fluxes(data, pos,prof=None,dy=5, method='box'):
    flux = np.empty(pos.shape)
    ### Loop over orders and wavlength channels for the specified fiber
    ### We can definitely find ways to speed up the nested loop
    if method=='box':
        for order in range(pos.shape[0]):
            for channel in range(pos.shape[1]):
                # print(pos[order, channel])
                y = int(np.round(pos[order, channel],0))
                ### Check axes with some simulated data
                ### Simple sum of the flux
                flux[order, channel] = np.sum(data[y-dy:y+dy+1,channel])
                ### Assuming the errors add in quadrature but can revisit
                # err[channel,order] = np.sqrt(np.sum(err[y-dy:y+dy+1,channel]**2))
                ### Assuming there are no covariances so that this is a straight sum
                # rnoise[channel,order] = np.sum(var_rnoise[y-dy:y+dy+1,channel])
                # poisson[channel,order] = np.sum(var_poisson[y-dy:y+dy+1,channel])
                ### Just check if any pixels in the slice were bad or do something more complicated?
                # dq = int(~np.any(dq[y-dy:y+dy+1,channel] != 0))
    elif method == 'optimal':
        ### Loop over pixels
        for xpix in range(pos.shape[1]):
            ### Get data slice around position of each order
            cutdata = np.zeros((prof.shape[0],prof.shape[-1]))
            cutprof = np.zeros((prof.shape[0],prof.shape[-1]))
            for i in range(prof.shape[0]):
                ### Cut data. Careful that this is done exactly how it was done when the 
                ### trace profile was calculated
                center = int(pos[i,xpix])
                width = (prof.shape[-1]-1) // 2
                dslice = data[center-width:center+width+1,xpix]
                pslice = np.atleast_2d(prof[i,xpix])
                ### Now we do the linear algebra to get the flux, solving
                ### data = flux x profile for the OLS flux value
                ### Did the linalg explicitly so that we could maybe
                ### jit compile this, and because scipy is super slow
                inv = np.linalg.inv(np.dot(pslice,pslice.T))
                pfactor = np.dot(pslice.T, inv)
                flux[i,xpix] = np.sum(dslice*pfactor[:,0])
                ### FIXME how does error propegation work with this

    return flux


if __name__ == '__main__':
    fname_trace = 'demo_frames/fixed_20231006B_HISPEC_SPECTRO_HK_pyechelle_flatsource_fiber2.fits'
    fnames_flux =  ['demo_frames/fixed_20231006B_HISPEC_SPECTRO_HK_pyechelle_flatsource_fiber2.fits']
    
    hdul_trace = fits.open(fname_trace)
    data_trace = hdul_trace[0].data
    trace_locs, _, trace_widths, _ = find_trace_locs(data_trace)
    trace_profile = fit_trace_prof(data_trace, trace_locs,7)

    ### Plot the trace locations
    # plt.imshow(data_trace,vmin=np.nanpercentile(data_trace,5),vmax=np.nanpercentile(data_trace,95))
    # for order in range(trace_locs.shape[0]):
    #     plt.plot(trace_locs[order,:],color='k')
    # plt.show()

    # ### Plot the rough measured FWHM
    # for order in range(trace_locs.shape[0]):
    #     plt.plot(trace_widths[order,:],color='k')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Trace FWHM')
    # plt.show()

    # ### Plot the trace profiles
    # ### Note that this is double peaked because of how things are rounded
    # ### that is fine as long as the optimal extraction code does the rounding
    # ### in the same way when it goes to fit things
    # for pix in range(trace_profile.shape[1]):
    #     plt.plot(trace_profile[4,pix,:],color='k',alpha=0.01)
    # plt.xlabel('Pixel from integer trace center')
    # plt.ylabel('Profile weight')
    # plt.show()


    for fname in fnames_flux:
        hdul_flux = fits.open(fname)
        data_flux = hdul_flux[0].data
        fluxes_box = extract_fluxes(data_flux, trace_locs, prof=trace_profile,method='box')
        np.save('demo_output/'+fname[12:-5]+'_1D_extracted_box.npy',fluxes_box)

        for row in fluxes_box:
            plt.plot(row)
        plt.xlabel('X Pixel')
        plt.ylabel('Box extracted flux')
        plt.show()

    for fname in fnames_flux:
        hdul_flux = fits.open(fname)
        data_flux = hdul_flux[0].data
        fluxes_opt = extract_fluxes(data_flux, trace_locs, prof=trace_profile,method='optimal')
        np.save('demo_output/'+fname[12:-5]+'_1D_extracted_optimal.npy',fluxes_opt)

        for row in fluxes_opt:
            plt.plot(row)
        plt.xlabel('X Pixel')
        plt.ylabel('Optimal extracted flux')
        plt.show()

    for row in range(fluxes_opt.shape[0]):
        plt.plot(100*(fluxes_opt[row] - fluxes_box[row])/fluxes_opt[row],color='k',alpha=0.05)
    plt.ylabel('Percent difference between box and optimal')
    plt.xlabel('Detector X Pixel')
    plt.show()

        