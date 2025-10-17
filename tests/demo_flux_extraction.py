import numpy as np
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt

def find_trace_locs(data):
    threshold_peak = 500
    cutsize = 10
    deg = 4

    ### Initial peak finding
    peaks,_ = signal.find_peaks(data[:,0],height=threshold_peak)
    
    ### Declare output arrays
    trace_locs = np.zeros((len(peaks),data.shape[1]))
    smooth_trace_locs = np.zeros(trace_locs.shape)
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
        peaks, _ = signal.find_peaks(data[:,i],height=threshold_peak)
        ### Get flux-weighted center for each peak on its own
        for j, peak in enumerate(peaks):
            inds = np.arange(peak-cutsize,peak+cutsize+1)
            dslice = data[inds,i]
            trace_locs[j,i] = np.sum(inds*dslice)/np.sum(dslice)

    ### Do polynomial fit to traces
    xpts = np.arange(trace_locs.shape[1])
    for i in range(trace_locs.shape[0]):
        ypts = trace_locs[i]
        coeffs[i] = np.polyfit(xpts, ypts, deg)
        fit = np.poly1d(coeffs[i])
        smooth_trace_locs[i,:] = fit(xpts)

    return smooth_trace_locs

def extract_fluxes(data, pos,dy=5):
    flux = np.empty(pos.shape)
    print(pos.shape)
    ### Loop over orders and wavlength channels for the specified fiber
    ### We can definitely find ways to speed up the nested loop
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
    return flux


if __name__ == '__main__':
    fname_trace = 'demo_frames/fixed_20231006B_HISPEC_SPECTRO_HK_pyechelle_flatsource_fiber2.fits'
    fnames_flux =  ['demo_frames/fixed_20231006B_HISPEC_SPECTRO_HK_pyechelle_flatsource_fiber2.fits']
    
    hdul_trace = fits.open(fname_trace)
    data_trace = hdul_trace[0].data
    trace_locs = find_trace_locs(data_trace)

    plt.imshow(data_trace,vmin=np.nanpercentile(data_trace,5),vmax=np.nanpercentile(data_trace,95))
    for order in range(trace_locs.shape[0]):
        plt.plot(trace_locs[order,:],color='k')
    plt.show()


    for fname in fnames_flux:
        hdul_flux = fits.open(fname)
        data_flux = hdul_flux[0].data
        fluxes = extract_fluxes(data_flux, trace_locs)
        np.save('demo_output/'+fname[12:-5]+'_1D_extracted.npy',fluxes)

        for row in fluxes:
            plt.plot(row)
        plt.show()

        