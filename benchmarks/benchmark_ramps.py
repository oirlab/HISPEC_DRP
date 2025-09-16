from hispecdrp import Stage0Pipeline
from hispecdrp import datamodels
import numpy as np
import time
import matplotlib.pyplot as plt

from memory_profiler import memory_usage

from utils import add_default_metadata


def create_ramp_model(slope, size=4096, n_reads=8, gain=2.28, rdnoise=12.0,
    seed=None):

    # Create a gain array with 2% error
    gain_img = np.random.normal(gain,0.02*gain,size*size).reshape((size,size))

    coeffs = np.broadcast_to(np.array([[[0]], [[1]]], dtype=np.float32), (2, 4096, 4096))
    thresh = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    max = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    nonlin_model = datamodels.HISPECNonlinearCorrectionModel(data=dict(coeffs=coeffs, thresh=thresh, max=max))
    nonlin_model.meta.spectrograph = 'BSPEC'
    datamodels.set_default_meta(nonlin_model)

    source = np.full((4096, 4096), slope, dtype=np.float32)
    ramp_data = datamodels.create_ramp(source, readtime=1.67, n_reads=n_reads, bias=0, read_noise=rdnoise, poisson_noise=False, nonlin_coeffs=None)
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramp_data['data'], dq=ramp_data['dq'], times=ramp_data['times']))
    ramp_model.meta.spectrograph = 'BSPEC'
    datamodels.set_default_meta(ramp_model)

    # Create a read noise array
    if seed is not None:
        seed = int(seed)

    noise_rng = np.random.default_rng(seed)
    rdnoise_img = noise_rng.normal(scale=rdnoise/gain, size=(size, size))
    
    return ramp_model, gain_img, rdnoise_img, nonlin_model


def benchmark_ramp(n_reads, algorithm, max_cores):

    slope = 10
    gain = 2.28
    rdnoise = 8.5
    size = 4096

    ramp_model, gain_img, rdnoise_img, nonlin_model = create_ramp_model(slope=slope, 
        size=4096, n_reads=n_reads, gain=gain, rdnoise=rdnoise, seed=1345898922)

    # Setup and call the stage 0 pipeline
    stage0_pipe = Stage0Pipeline()

    # Set any primitive args
    stage0_pipe.nonlin_corr.nonlin = nonlin_model
    stage0_pipe.ramp_fit.method = algorithm
    stage0_pipe.ramp_fit.max_cores = max_cores

    # Run pipeline
    model_result = stage0_pipe.apply(ramp_model, output_dir='.')

def benchmark_ramp_fit(algorithm='jwst_likely', max_cores=8):

    outdata = []
    for i in np.arange(7):
        if i<2: continue
        n_reads = 2**i
        exptime = 1.67 * n_reads

        start_time = time.time()
        # Returns memory usage in MB in increments of 0.1s
        mem_usage = memory_usage((benchmark_ramp, [n_reads, algorithm, max_cores]))
        end_time = time.time()
        # Total run time in seconds
        runtime = end_time - start_time

        # Should be maximum memory usage in MB
        max_memory = np.max(mem_usage)

        outdata.append((n_reads, exptime, runtime, max_memory))

    fig, ax = plt.subplots()
    exptimes = [o[1] for o in outdata]
    runtimes = [o[2] for o in outdata]

    ax.scatter(exptimes, runtimes, marker='o')
    ax.set_xlabel('Exposure Times [s]', fontsize=16)
    ax.set_ylabel('Run Times [s]', fontsize=16)
    ax.set_title(f'algorithm={algorithm}, max_cores={max_cores}, Run Time', fontsize=12)
    plt.savefig(f'{algorithm}_runtimes.png')

    fig, ax = plt.subplots()
    mem_usg = [o[3]/1024. for o in outdata]
    ax.set_xlabel('Exposure Times [s]', fontsize=16)
    ax.set_ylabel('Memory Usage [GB]', fontsize=16)
    ax.scatter(exptimes, mem_usg, marker='o')
    ax.set_title(f'algorithm={algorithm}, max_cores={max_cores}, Memory', fontsize=12)
    plt.savefig(f'{algorithm}_mem_usg.png')


if __name__=="__main__":
    benchmark_ramp_fit(algorithm='jwst_likely', max_cores=8)
    benchmark_ramp_fit(algorithm='jwst_ols', max_cores=8)
