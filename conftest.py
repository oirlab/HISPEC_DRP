import glob
import os

import pytest

# Stray pipeline products that some tests write into the process cwd (often the repo root).
_REPO_ROOT_STRAY_GLOBS = (
    "HB.*.fits",
    "HR.*.fits",
    "simulated_ramp.fits",
    "rate_map.fits",
)


def _cleanup_repo_root_stray_fits() -> None:
    """Remove known test-generated FITS left in the repository root."""
    root = os.path.dirname(os.path.abspath(__file__))
    for pattern in _REPO_ROOT_STRAY_GLOBS:
        for path in glob.glob(os.path.join(root, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


def pytest_addoption(parser):
    parser.addoption(
        "--datadir",
        action="store",
        default=None,
        help="Directory to store test data"
    )
    parser.addoption(
        "--bspec-all-orders",
        action="store_true",
        default=False,
        help=(
            "BSPEC BT-Settl line-mask tests: use every echelle order from HISPECConfig "
            "(sets HISPEC_DRP_BSPEC_BTSETTL_ORDER=all). Same as exporting that env var."
        ),
    )
    parser.addoption(
        "--bspec-window-um",
        action="store",
        default=None,
        metavar="UM",
        help=(
            "BSPEC BT-Settl test_fit_ccf_tophat_style_btsettl_bspec_all_chunks_parallel: set "
            "window_size_um in micron (sets HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM). Default 0.3 "
            "when unset."
        ),
    )
    parser.addoption(
        "--bspec-relax-parallel-asserts",
        action="store_true",
        default=False,
        help=(
            "BSPEC BT-Settl test_fit_ccf_tophat_style_btsettl_bspec_all_chunks_parallel: skip strict "
            "template-CCF recovery, RVCCFAVG, and chunk-benchmark timing asserts "
            "(sets HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS=1). Use with non-default "
            "--bspec-window-um or when you need previews despite loose RV agreement."
        ),
    )


def pytest_configure(config):
    if config.getoption("--bspec-all-orders", default=False):
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_ORDER"] = "all"
    w_um = config.getoption("--bspec-window-um", default=None)
    if w_um is not None:
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM"] = str(w_um).strip()
    if config.getoption("--bspec-relax-parallel-asserts", default=False):
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS"] = "1"


def pytest_sessionfinish(session, exitstatus):
    # Always scrub repo-root litter, including interrupted / failed sessions.
    _cleanup_repo_root_stray_fits()


@pytest.fixture(scope="session")
def datadir(request):
    user_dir = request.config.getoption("--datadir")
    if user_dir:
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    # fallback: use a default location in the repo or a temp dir
    #default = os.path.abspath("test_data_cache")
    #os.makedirs(default, exist_ok=True)
    return None