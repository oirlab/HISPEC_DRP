"""CLI mode wiring for :mod:`hispecdrp.simulations.run_e2e_sim` (no full E2E run)."""

from __future__ import annotations

from hispecdrp.simulations import run_e2e_sim


def test_run_e2e_sim_help_exits_zero():
    try:
        run_e2e_sim.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_base_config_defaults():
    cfg = run_e2e_sim._base_config(output_dir="/tmp/e2e_test_out")
    assert cfg.output_dir == "/tmp/e2e_test_out"
    assert cfg.spectrograph == "BSPEC"
    assert cfg.gaia_source_id == run_e2e_sim.DEFAULT_GAIA_SOURCE_ID
    assert cfg.template_filename == run_e2e_sim.DEFAULT_TEMPLATE
    assert cfg.calibrations_only is False
    assert cfg.infinite_snr_rv_validation is False


def test_calibrations_only_mode_sets_flags(monkeypatch):
    calls = {}

    def fake_run_e2e_calibrations(config):
        calls["config"] = config

        class _Summary:
            rows = []
            validation_warnings = []

        class _Res:
            summary = _Summary()
            output_dir = config.output_dir

        return _Res()

    monkeypatch.setattr(run_e2e_sim, "run_e2e_calibrations", fake_run_e2e_calibrations)
    rc = run_e2e_sim.main(["--calibrations-only", "--output-dir", "/tmp/e2e_cals"])
    assert rc == 0
    cfg = calls["config"]
    assert cfg.calibrations_only is True
    assert cfg.run_stage1 is False
    assert cfg.run_stage2 is False
    assert cfg.write_calibration_manifest is True


def test_infinite_snr_mode_applies_validation_flag(monkeypatch):
    calls = {}

    def fake_apply(config):
        calls["applied"] = True
        config.infinite_snr_rv_validation = True
        return config

    def fake_run_e2e_simulation(config):
        calls["config"] = config

        class _Res:
            stage2_model = None
            infinite_snr_rv_evaluation = None

        return _Res()

    monkeypatch.setattr(
        "hispecdrp.simulations.e2e_pipeline.apply_infinite_snr_rv_e2e_config",
        fake_apply,
    )
    monkeypatch.setattr(run_e2e_sim, "run_e2e_simulation", fake_run_e2e_simulation)
    rc = run_e2e_sim.main(["--infinite-snr-rv", "--output-dir", "/tmp/e2e_base"])
    assert rc == 0
    assert calls.get("applied") is True
    assert calls["config"].infinite_snr_rv_validation is True
    assert calls["config"].output_dir.endswith("infinite_snr_rv")


def test_removed_thin_cli_modules_are_gone():
    import importlib.util

    for name in (
        "hispecdrp.simulations.run_e2e_cals",
        "hispecdrp.simulations.run_e2e_infinite_snr_rv",
        "hispecdrp.simulations.e2e_infinite_snr_rv",
        "hispecdrp.simulations.sim_hdf_calibrations",
        "hispecdrp.simulations.e2e_previews",
    ):
        assert importlib.util.find_spec(name) is None, f"{name} should have been removed"
