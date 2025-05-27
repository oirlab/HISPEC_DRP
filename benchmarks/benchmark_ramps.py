from hispecdrp import FitRampPrimitive
import textwrap
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

    # Create a read noise array
    if seed is not None:
        seed = int(seed)

    noise_rng = np.random.default_rng(seed)
    rdnoise_img = noise_rng.normal(scale=rdnoise/gain, size=(size, size))

    # Create ramp
    times = np.arange(n_reads).astype(np.float64)
    ramps = np.zeros(shape=(size, size, n_reads), dtype=np.float32)
    for i in range(n_reads):
        ramps[:, :, i] = slope/gain * times[i] + noise_rng.normal(scale=rdnoise/gain, size=(size, size))
    ramp_model = datamodels.HISPECRampModel(data=dict(data=ramps, times=times))
    add_default_metadata(ramp_model)
    
    return ramp_model, gain_img, rdnoise_img


def benchmark_ramp(n_reads):

    slope = 10
    gain = 2.28
    rdnoise = 8.5
    size = 4096

    ramp_model, gain_img, rdnoise_img = create_ramp_model(slope=slope, 
        size=4096, n_reads=n_reads, gain=gain, rdnoise=rdnoise, seed=1345898922)

    # Config file
    rampfit_config_file = str('ramp_config.yaml')
    rampfit_config_raw = """
    %YAML 1.2
    ---

    logging:
        enable: True
        level: info
        verbose: True
        log_path: None
        
    arguments:
        save_results: False
        suffix: 'ramp-fit'
    """
    with open(rampfit_config_file, 'w') as f:
        f.write(textwrap.dedent(rampfit_config_raw))

    # Setup and call the flat primitive
    rampfit_prim = FitRampPrimitive(config_file=rampfit_config_file)
    model_result = rampfit_prim.apply(ramp_model, output_dir='.', method='jwst_likely', max_cores=8)

def benchmark_ramp_fit():

    outdata = []
    for i in np.arange(7):
        if i<2: continue
        n_reads = 2**i
        exptime = 1.67 * n_reads

        start_time = time.time()
        # Returns memory usage in MB in increments of 0.1s
        mem_usage = memory_usage((benchmark_ramp, [n_reads]))
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
    ax.set_xlabel('Exposure Times [s]')
    ax.set_ylabel('Run Times [s]')
    plt.savefig('runtimes.png')

    fig, ax = plt.subplots()
    mem_usg = [o[3]/1024. for o in outdata]
    ax.set_xlabel('Exposure Times [s]')
    ax.set_ylabel('Memory Usage [GB]')
    ax.scatter(exptimes, mem_usg, marker='o')
    plt.savefig('mem_usg.png')


if __name__=="__main__":
    benchmark_ramp_fit()
