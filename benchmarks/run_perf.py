
from .stage0_perf import run_perf_stage0
from .stage1_perf import run_perf_stage1
from .stage2_perf import run_perf_stage2

import os

if __name__ == "__main__":

    n_reads = 20 # 350
    max_cores = 4
    output_dir = os.path.abspath('test_data/')
    quicklook = False
    
    run_perf_stage0(
        output_dir=output_dir,
        n_reads=n_reads,
        max_cores=max_cores,
        save_result=True,
        quicklook=quicklook
    )

    run_perf_stage1(
        output_dir=output_dir,
        max_cores=max_cores,
        save_result=True,
        quicklook=quicklook
    )

    run_perf_stage2(
        output_dir=output_dir,
        max_cores=max_cores,
        save_result=True,
        do_fibers=['1'],
        quicklook=quicklook
    )