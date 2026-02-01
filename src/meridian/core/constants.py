"""Application constants."""

from enum import Enum


class CustomerSegment(str, Enum):
    """Customer segment types."""

    HIGH_VALUE = "high_value"
    MEDIUM_VALUE = "medium_value"
    LOW_VALUE = "low_value"
    NEW = "new"
    AT_RISK = "at_risk"
    CHURNED = "churned"


class ExperimentStatus(str, Enum):
    """Experiment status types."""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    """ML model types."""

    CAUSAL_FOREST = "causal_forest"
    X_LEARNER = "x_learner"
    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"
    DEEPAR = "deepar"
    NBEATS = "nbeats"
    ELASTICITY = "elasticity"


class Channel(str, Enum):
    """Customer channel types."""

    ONLINE = "online"
    STORE = "store"
    MOBILE = "mobile"
    CALL_CENTER = "call_center"


# API limits
MAX_BATCH_SIZE = 10000
MAX_CUSTOMER_IDS_PER_REQUEST = 1000
MAX_FORECAST_HORIZON_DAYS = 365

# Model defaults
DEFAULT_UPLIFT_MODEL = "causal_forest_v1"
DEFAULT_FORECAST_MODEL = "deepar_v1"
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Cache TTL (seconds)
PREDICTION_CACHE_TTL = 3600  # 1 hour
FEATURE_CACHE_TTL = 300  # 5 minutes
MODEL_CACHE_TTL = 86400  # 24 hours

