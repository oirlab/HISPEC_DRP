import numpy as np
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt
from demo_flux_extraction import find_trace_locs

if __name__ == '__main__':
	fname = 'demo_frames/fixed_20231006B_HISPEC_SPECTRO_HK_pyechelle_flatsource_fiber2.fits'
	hdul = fits.open(fname)
	data = hdul[0].data

	
	smooth_trace_locs = find_trace_locs(data)


	plt.imshow(data,vmin=np.nanpercentile(data,5),vmax=np.nanpercentile(data,95))
	for order in range(smooth_trace_locs.shape[0]):
		plt.plot(smooth_trace_locs[order,:],color='k')
	plt.show()
