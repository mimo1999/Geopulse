"""PyTorch model package for geopolitical risk forecasting."""
from models.risk_model import HybridRiskTransformer   # no DB dependency

# DB-dependent classes are imported lazily to allow training without psycopg2
def _lazy_dataset():
    from models.dataset import CountryRiskDataset, RiskDataLoader
    return CountryRiskDataset, RiskDataLoader

__all__ = ["HybridRiskTransformer"]
