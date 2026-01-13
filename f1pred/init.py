# Import submodules so they are available at the package top level
# Removed 'report' as it has been deleted from the codebase
from . import config, util, data, features, models, simulate, ranking, predict, backtest, live

# metrics may not exist in all revisions; try to import it but don't fail the package if it's missing
try:
    from . import metrics
    __all__ = [
        "config", "util", "data", "features", "models",
        "simulate", "ranking", "predict", "backtest",
        "live", "metrics",
    ]
except ImportError:
    __all__ = [
        "config", "util", "data", "features", "models",
        "simulate", "ranking", "predict", "backtest",
        "live",
    ]
