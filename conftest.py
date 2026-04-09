import pytest
import os

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
            "BSPEC BT-Settl mask-KPF tests: use every echelle order from HISPECConfig "
            "(sets HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1). Same as exporting that env var."
        ),
    )


def pytest_configure(config):
    if config.getoption("--bspec-all-orders", default=False):
        os.environ["HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS"] = "1"

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