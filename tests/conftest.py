import glob
import os
from unittest import mock

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
    # conftest lives in tests/; repo root is one level up
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        "--cal-db-dir",
        action="store",
        default=None,
        help=(
            "Directory for the shared simulated cal DB (built for tests marked cal_db). It is kept after the "
            "run and reused while complete. Default: <pytest cache dir>/d/hispec_cal_db."
        ),
    )
    parser.addoption(
        "--remake-cal-db",
        action="store_true",
        default=False,
        help="Rebuild the shared simulated cal DB (for tests marked cal_db) even if a complete one exists.",
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
    config.addinivalue_line(
        "markers",
        "cal_db: test needs the shared simulated cal DB; it is built before the test if missing.",
    )
    if config.getoption("--bspec-all-orders", default=False):
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_ORDER"] = "all"
    w_um = config.getoption("--bspec-window-um", default=None)
    if w_um is not None:
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM"] = str(w_um).strip()
    if config.getoption("--bspec-relax-parallel-asserts", default=False):
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS"] = "1"


def _cal_db_root(config) -> str:
    root = config.getoption("--cal-db-dir")
    if root is None:
        root = str(config.cache.mkdir("hispec_cal_db"))
    return os.path.abspath(root)


def pytest_sessionstart(session):
    """Point the calibration store at the shared simulated cal DB, with remote KOA off.

    Every test then finds the shared cals through the store (e.g. ``Primitive.get_calibration``).
    The DB itself is only built for tests marked ``cal_db`` (see `pytest_runtest_setup`).
    Tests that register cals themselves use ``standalone_cal_cache``.
    """
    from tests.sim_cals import CAL_DB_CACHE

    root = _cal_db_root(session.config)
    os.makedirs(root, exist_ok=True)
    os.environ["KOA_CALIBRATION_CACHE"] = os.path.join(root, CAL_DB_CACHE) + os.sep
    os.environ["HISPEC_KOA_CONNECT_REMOTE"] = "false"


_CAL_DB_READY = False


def pytest_runtest_setup(item):
    """Before the first test marked ``cal_db``, build the shared cal DB if it is missing or incomplete.

    The DB is kept after the run and reused while complete (see ``tests/sim_cals.py`` for what is
    built). Use --remake-cal-db to rebuild it and --cal-db-dir to choose where it lives.
    """
    global _CAL_DB_READY
    if _CAL_DB_READY or item.get_closest_marker("cal_db") is None:
        return
    from tests.sim_cals import load_cal_db, make_cal_db

    root = _cal_db_root(item.config)
    if item.config.getoption("--remake-cal-db") or load_cal_db(root) is None:
        make_cal_db(root)
    _CAL_DB_READY = True


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


@pytest.fixture
def standalone_cal_cache(tmp_path):
    """Point KOA_CALIBRATION_CACHE at a fresh, empty cache under tmp_path for one test.

    Only for tests that register calibrations themselves; the previous value is restored after.
    """
    cache = tmp_path / "KOA_CALIBRATION_CACHE"
    cache.mkdir()
    with mock.patch.dict(os.environ, {"KOA_CALIBRATION_CACHE": str(cache) + os.sep}):
        yield cache