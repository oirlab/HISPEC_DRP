from hispecdrp import ProcessBiasPipeline
from hispecdrp import datamodels
import hispecdrp.simulations as simu
import numpy as np
import os
import time
from hispecdrp.primitives.bias_subtraction.master_bias_primitive import MakeMasterBiasPrimitive
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_bias_calibration_results(
    bias_true, bias_measured, ktc_true, ktc_measured,
    residuals_bias, residuals_ktc,
    output_dir=None, show=True
):
    """
    Create Plotly visualizations of bias calibration results.
    
    Parameters
    ----------
    bias_true : np.ndarray
        True bias map in DN
    bias_measured : np.ndarray
        Measured bias map in DN
    ktc_true : np.ndarray
        True kTC map in DN
    ktc_measured : np.ndarray
        Measured kTC map in DN
    residuals_bias : np.ndarray
        Residuals for bias in DN
    residuals_ktc : np.ndarray
        Residuals for kTC in DN
    output_dir : str, optional
        Directory to save plots. If None, plots are not saved.
    show : bool, optional
        Whether to display the plots. Default is True.
    
    Returns
    -------
    dict
        Dictionary containing figure objects with keys 'bias' and 'ktc'
    """
    
    ny, nx = bias_true.shape
    
    # Create subplots for bias
    fig_bias = make_subplots(
        rows=1, cols=3,
        subplot_titles=("True Bias (DN)", "Measured Bias (DN)", "Residuals (DN)"),
        specs=[[{}, {}, {}]]
    )
    
    # True bias
    fig_bias.add_trace(
        go.Heatmap(
            z=bias_true,
            colorscale='Inferno',
            showscale=True,
            name='True',
            colorbar=dict(x=0.16, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=1
    )
    
    # Measured bias
    fig_bias.add_trace(
        go.Heatmap(
            z=bias_measured,
            colorscale='Viridis',
            showscale=True,
            name='Measured',
            colorbar=dict(x=0.5, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=2
    )
    
    # Residuals bias
    fig_bias.add_trace(
        go.Heatmap(
            z=residuals_bias,
            colorscale='RdBu',
            showscale=True,
            name='Residuals',
            colorbar=dict(x=0.84, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=3
    )
    
    fig_bias.update_xaxes(showgrid=False)
    fig_bias.update_yaxes(showgrid=False)
    
    fig_bias.update_layout(
        height=650,
        width=1400,
        title_text="Bias Calibration Results",
        showlegend=False,
        margin=dict(b=180)
    )
    
    # Create subplots for kTC
    fig_ktc = make_subplots(
        rows=1, cols=3,
        subplot_titles=("True kTC (DN)", "Measured kTC (DN)", "Residuals (DN)"),
        specs=[[{}, {}, {}]]
    )
    
    # True kTC
    fig_ktc.add_trace(
        go.Heatmap(
            z=ktc_true,
            colorscale='Viridis',
            showscale=True,
            name='True',
            colorbar=dict(x=0.16, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=1
    )
    
    # Measured kTC
    fig_ktc.add_trace(
        go.Heatmap(
            z=ktc_measured,
            colorscale='Viridis',
            showscale=True,
            name='Measured',
            colorbar=dict(x=0.5, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=2
    )
    
    # Residuals kTC
    fig_ktc.add_trace(
        go.Heatmap(
            z=residuals_ktc,
            colorscale='RdBu',
            showscale=True,
            name='Residuals',
            colorbar=dict(x=0.84, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
        ),
        row=1, col=3
    )
    
    fig_ktc.update_xaxes(showgrid=False)
    fig_ktc.update_yaxes(showgrid=False)
    
    fig_ktc.update_layout(
        height=650,
        width=1400,
        title_text="kTC Noise Calibration Results",
        showlegend=False,
        margin=dict(b=180)
    )
    
    # Save if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        bias_html = os.path.join(output_dir, 'bias_calibration_preview.html')
        ktc_html = os.path.join(output_dir, 'ktc_calibration_preview.html')
        
        fig_bias.write_html(bias_html, include_plotlyjs='cdn')
        fig_ktc.write_html(ktc_html, include_plotlyjs='cdn')
        
        print(f"Bias calibration plots saved to {bias_html}")
        print(f"kTC calibration plots saved to {ktc_html}")
    
    if show:
        fig_bias.show()
        fig_ktc.show()
    
    return {
        'bias': fig_bias,
        'ktc': fig_ktc
    }


def test_stage0_bias_calib(tmp_path):
    np.random.seed(12345)
    start_time = time.time()

    # --- Define parameters for the simulated ramp ---
    max_cores = 4
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 1
    N_files = 10 # number of simulated files to create
    ny, nx = 4096,4096  # size of the detector

    step_time = time.time();
    print(f"Overheads, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, 1.5 * np.iinfo(np.uint16).max, 1001, endpoint=True)
    # nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    nonlin_coeffs = [1, 0]  # Example coefficients for a cubic non-linearity
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)
    print("Fitted correction coefficients:", nonlin_correction_coefs)

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, nonlinear_ramp)

    coeffs = np.broadcast_to(nonlin_correction_coefs[::-1, None, None], (np.size(nonlin_correction_coefs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.full(shape=(ny, nx), fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
    nonlin_dq = np.zeros(shape=(ny, nx), dtype=np.uint32)
    nonlin_model = datamodels.NonlinearCorrectionModel(
        coeffs=coeffs,
        nonlin_thresh=thresh,
        nonlin_max=max,
        dq=nonlin_dq
    )

    step_time = time.time();
    print(f"Step: defining nonlin coefficients, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create DQ init map ---
    dqinit_map = np.zeros((ny, nx), dtype=np.uint32)
    # set row to bad
    # dqinit_map[5, :] |= DQ_FLAGS["DO_NOT_USE"]
    dqinit_model = datamodels.DQModel(dq=dqinit_map)

    step_time = time.time();
    print(f"Step: defining DQ init calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(
        sat_thresh=saturation_map,
        dq=np.zeros_like(saturation_map, dtype=np.uint32)
    )

    step_time = time.time();
    print(f"Step: defining saturation calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random gain map ---
    gain = 3  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(
        gain=gain_map,
        rn=np.full_like(gain_map, 0.0),
        cov=np.full((2, 2, ny, nx), 0, dtype=np.float32),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random dark map ---
    dark_current = 5  # e-/s
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    # dark_map = np.ones((ny, nx), dtype=np.float32)
    dark_model = datamodels.DarkModel(
        rate=dark_map / gain_map,  # dark in DN/s
        rate_err=np.zeros_like(dark_map),
        rate_var_rnoise=np.zeros_like(dark_map),
        rate_var_poisson=np.zeros_like(dark_map),
        dq=np.zeros_like(dark_map, dtype=np.uint32)
    )

    step_time = time.time();
    print(f"Step: defining dark calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create bias map ---
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    kTC_noise = 50  # e-/s
    kTC_noise_map = np.full((ny, nx), kTC_noise, dtype=np.float32)
    bias_model = datamodels.BiasModel(
        bias=bias_map / gain_map,  # bias in DN
        ktc=kTC_noise_map / gain_map,  # kTC in DN
        dq=np.zeros_like(bias_map, dtype=np.uint32)
    )

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random flat map ---
    # detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_map = np.ones((ny, nx), dtype=np.float32)
    detflat_model = datamodels.DetectorFlatModel(
        flat=detflat_map,
        flat_err=np.full_like(gain_map, 0.03),
        flat_var_rnoise=np.full_like(gain_map, 0.01),
        flat_var_poisson=np.full_like(gain_map, 0.02),
        dq=np.zeros_like(detflat_map, dtype=np.uint32)
    )

    step_time = time.time();
    print(f"Step: defining flat calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random read noise map ---
    RN = 8.5  # e-
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(
        rn=rn_map / gain_map,
        rn_err=np.ones_like(rn_map),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full((ny, nx), max_slope, dtype=np.float32)
    start_time = time.time()
    ramp_model_sequence = []
    for k in range(N_files):
        ramp_data = simu.create_ramp(
            electron_rate_map,  # in e-/s
            readtime=readtime, n_reads=n_reads,
            nonlin_coeffs=nonlin_coeffs,
            gain=gain_map,  # in e-/s
            flat=detflat_map,
            dark=dark_map,  # in e-/s
            bias=bias_map,  # in e-/s
            kTC_noise=kTC_noise,  # in e-/s
            poisson_noise=True, read_noise=rn_map,  # in e-/s
            convert_to_uint16=True, clip_ramps=True,
            max_cores=max_cores
        )
        # adding a jump
        # ramp_data['data'][4::,:,:] += 1000
        # ramp_data['data'][10::,:,:] += 1000
        # ramp_data['data'][15::,:,:] += 1000
        # ramp_data['data'][20::,:,:] += 1000
        # ramp_data['data'][4::,5,5] += 3000

        # print(ramp_data['data'][:,0,0])
        # print(gain_map)
        # exit()

        step_time = time.time();
        print(f"Step: creating ramp (new), elapsed: {step_time - start_time:.2f}s")
        start_time = time.time()

        ramp_model = datamodels.RampModel(
            data=ramp_data['data'],
            dq_raw=ramp_data['dq_raw'],
            meta=ramp_data['meta']
        )
        ramp_model.meta.filename = "simulated_ramp.fits"
        print(ramp_model.meta.clock_rate)

        ramp_model_sequence.append(ramp_model)
    # exit()

    step_time = time.time();
    print(f"Step: creating ramp, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    pipe_max_cores = max_cores
    # --- Specify the primitive arguments here ---
    prim_args = {
        'dq_init': {
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'make_bias': {
            'run': True,
            'max_cores': pipe_max_cores,
            'save_preview': True,
            'show_preview': True,
        },
        'register_bias_cal': {
            'run': True,
        }
    }

    # Setup and call the stage 0 pipeline
    bias_pipe = ProcessBiasPipeline()

    # Enable preview generation
    bias_pipe.save_preview = True
    bias_pipe.show_preview = True  
    bias_pipe.output_dir = str(tmp_path)

    # Set any primitive args
    bias_pipe.dq_init.dq = dqinit_model

    step_time = time.time();
    print(f"Step: defining pipeline, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # Run pipeline
    model_result = bias_pipe.apply(ramp_model_sequence, prim_args=prim_args)

    # print("Bias")
    # print(model_result['bias'].bias)
    # print("kTC")
    # print(model_result['bias'].kTC)

    # Residuals (bias - true bias)
    residuals_bias = np.abs(model_result['bias'].bias - bias_map / gain_map) # in DN
    
    # Residuals (kTC - true kTC)
    residuals_kTC = np.abs(model_result['bias'].ktc - kTC_noise_map / gain_map) # in DN
    
    # # Create visualizations
    # plots = plot_bias_calibration_results(
    #     bias_true=bias_map / gain_map,
    #     bias_measured=model_result['bias'].bias,
    #     ktc_true=kTC_noise_map / gain_map,
    #     ktc_measured=model_result['bias'].ktc,
    #     residuals_bias=residuals_bias,
    #     residuals_ktc=residuals_kTC,
    #     output_dir=str(tmp_path),
    #     show=True  # Set to True if you want interactive display
    # )
    try:
        assert np.sum(residuals_bias > 10) == 0
    except AssertionError:
        print("Bias residuals exceed threshold at the following pixels:")
        print(np.argwhere(residuals_bias > 10))

    # Residuals (kTC - true kTC)
    
    try:
        assert np.sum(residuals_kTC > 5) == 0
    except AssertionError:
        print("kTC residuals exceed threshold at the following pixels:")
        print(np.argwhere(residuals_kTC > 5))



def test_stage0_master_bias(tmp_path):
    tmp_path = str(tmp_path)
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # --- Create bias map ---
    ny, nx = 10,10  # size of the detector
    gain = 1  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    kTC_noise = 50  # e-/s
    kTC_noise_map = np.full((ny, nx), kTC_noise, dtype=np.float32)
    bias_model1 = datamodels.BiasModel(
        bias=bias_map / gain_map,  # bias in DN
        ktc=kTC_noise_map / gain_map,  # kTC in DN
        dq=np.zeros_like(bias_map, dtype=np.uint32)
    )
    bias_model2 = bias_model1.copy()
    bias_model3 = bias_model1.copy()
    bias_model4 = bias_model1.copy()

    bias_list = [bias_model1, bias_model2, bias_model3, bias_model4]

    make_bias_primitive = MakeMasterBiasPrimitive()

    output_master_bias = make_bias_primitive.apply(bias_list)

    assert  output_master_bias.meta.master_cal
    np.testing.assert_array_equal(output_master_bias.bias, bias_map / gain_map)
    np.testing.assert_array_equal(output_master_bias.kTC, kTC_noise_map)  # error should reduce by sqrt(4)

if __name__ == "__main__":
    test_stage0_bias_calib(tmp_path='/Users/bensappey/scratch/')
    