import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "cuda: mark that test requires CUDA"
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA support not enabled")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
