import matplotlib
import pytest

# Force non-interactive backend for tests
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test to free memory."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
