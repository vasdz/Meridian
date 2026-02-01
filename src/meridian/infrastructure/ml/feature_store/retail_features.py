"""Feature engineering for retail customer analytics.

Includes:
- RFM (Recency, Frequency, Monetary) features
- Behavioral patterns
- External factors (seasonality, holidays)
- Communication channel features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from meridian.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RFMFeatures:
    """RFM feature container."""

    recency: float  # Days since last purchase
    frequency: int  # Number of purchases
    monetary: float  # Total spending

    # Derived features
    recency_score: int  # 1-5 score
    frequency_score: int  # 1-5 score
    monetary_score: int  # 1-5 score
    rfm_score: str  # Combined score e.g., "543"
    rfm_segment: str  # Segment name

    def to_dict(self) -> dict:
        return {
            "recency": self.recency,
            "frequency": self.frequency,
            "monetary": self.monetary,
            "recency_score": self.recency_score,
            "frequency_score": self.frequency_score,
            "monetary_score": self.monetary_score,
            "rfm_score": self.rfm_score,
            "rfm_segment": self.rfm_segment,
        }


class RFMCalculator:
    """Calculate RFM features for customers."""

    # RFM segments mapping
    SEGMENTS = {
        "Champions": ["555", "554", "544", "545", "454", "455", "445"],
        "Loyal": ["543", "444", "435", "355", "354", "345", "344", "335"],
        "Potential Loyalists": ["553", "551", "552", "541", "542", "533", "532", "531", "452", "451", "442", "441", "431", "453", "433", "432", "423", "353", "352", "351", "342", "341", "333", "323"],
        "New Customers": ["512", "511", "422", "421", "412", "411", "311"],
        "Promising": ["525", "524", "523", "522", "521", "515", "514", "513", "425", "424", "413", "414", "415", "315", "314", "313"],
        "Need Attention": ["535", "534", "443", "434", "343", "334", "325", "324"],
        "About to Sleep": ["331", "321", "312", "221", "213", "231", "241", "251"],
        "At Risk": ["255", "254", "245", "244", "253", "252", "243", "242", "235", "234", "225", "224", "153", "152", "145", "143", "142", "135", "134", "133", "125", "124"],
        "Hibernating": ["155", "154", "144", "214", "215", "115", "114", "113"],
        "Lost": ["111", "112", "121", "131", "141", "151", "211", "212", "222", "223", "232", "233"],
    }

    def __init__(
        self,
        recency_bins: int = 5,
        frequency_bins: int = 5,
        monetary_bins: int = 5,
    ):
        self.recency_bins = recency_bins
        self.frequency_bins = frequency_bins
        self.monetary_bins = monetary_bins

        # Thresholds (will be fitted)
        self._recency_thresholds: Optional[np.ndarray] = None
        self._frequency_thresholds: Optional[np.ndarray] = None
        self._monetary_thresholds: Optional[np.ndarray] = None

    def fit(
        self,
        recency: np.ndarray,
        frequency: np.ndarray,
        monetary: np.ndarray,
    ) -> "RFMCalculator":
        """Fit RFM thresholds based on data distribution."""

        # Calculate quantile thresholds
        self._recency_thresholds = np.percentile(
            recency,
            np.linspace(0, 100, self.recency_bins + 1)[1:-1]
        )
        self._frequency_thresholds = np.percentile(
            frequency,
            np.linspace(0, 100, self.frequency_bins + 1)[1:-1]
        )
        self._monetary_thresholds = np.percentile(
            monetary,
            np.linspace(0, 100, self.monetary_bins + 1)[1:-1]
        )

        logger.info(
            "RFM thresholds fitted",
            recency_thresholds=self._recency_thresholds.tolist(),
            frequency_thresholds=self._frequency_thresholds.tolist(),
            monetary_thresholds=self._monetary_thresholds.tolist(),
        )

        return self

    def _score_recency(self, recency: float) -> int:
        """Score recency (lower is better, so reverse scoring)."""
        if self._recency_thresholds is None:
            raise RuntimeError("Fit the calculator first")

        # Reverse: low recency = high score
        score = self.recency_bins - np.searchsorted(self._recency_thresholds, recency)
        return max(1, min(self.recency_bins, score))

    def _score_frequency(self, frequency: int) -> int:
        """Score frequency (higher is better)."""
        if self._frequency_thresholds is None:
            raise RuntimeError("Fit the calculator first")

        score = np.searchsorted(self._frequency_thresholds, frequency) + 1
        return max(1, min(self.frequency_bins, score))

    def _score_monetary(self, monetary: float) -> int:
        """Score monetary (higher is better)."""
        if self._monetary_thresholds is None:
            raise RuntimeError("Fit the calculator first")

        score = np.searchsorted(self._monetary_thresholds, monetary) + 1
        return max(1, min(self.monetary_bins, score))

    def _get_segment(self, rfm_score: str) -> str:
        """Map RFM score to segment name."""
        for segment, scores in self.SEGMENTS.items():
            if rfm_score in scores:
                return segment
        return "Other"

    def calculate(
        self,
        recency: float,
        frequency: int,
        monetary: float,
    ) -> RFMFeatures:
        """Calculate RFM features for a single customer."""

        r_score = self._score_recency(recency)
        f_score = self._score_frequency(frequency)
        m_score = self._score_monetary(monetary)

        rfm_score = f"{r_score}{f_score}{m_score}"
        segment = self._get_segment(rfm_score)

        return RFMFeatures(
            recency=recency,
            frequency=frequency,
            monetary=monetary,
            recency_score=r_score,
            frequency_score=f_score,
            monetary_score=m_score,
            rfm_score=rfm_score,
            segment=segment,
        )

    def calculate_batch(
        self,
        recency: np.ndarray,
        frequency: np.ndarray,
        monetary: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate RFM features for multiple customers."""

        results = []
        for r, f, m in zip(recency, frequency, monetary):
            rfm = self.calculate(r, f, m)
            results.append(rfm.to_dict())

        return pd.DataFrame(results)


class BehavioralFeatureExtractor:
    """Extract behavioral patterns from transaction history."""

    def __init__(self):
        self.feature_names = [
            "avg_days_between_visits",
            "std_days_between_visits",
            "trend_days_between_visits",
            "avg_basket_size",
            "std_basket_size",
            "preferred_day_of_week",
            "preferred_hour",
            "weekend_ratio",
            "morning_ratio",
            "evening_ratio",
            "category_diversity",
            "brand_loyalty",
            "promo_sensitivity",
            "days_since_first_purchase",
            "purchase_velocity",
            "avg_discount_rate",
            "return_rate",
        ]

    def extract_from_transactions(
        self,
        transactions: pd.DataFrame,
        customer_id: str,
        date_col: str = "transaction_date",
        amount_col: str = "amount",
        category_col: Optional[str] = "category",
        is_promo_col: Optional[str] = "is_promo",
        discount_col: Optional[str] = "discount_amount",
    ) -> dict[str, float]:
        """
        Extract behavioral features from transaction history.

        Args:
            transactions: DataFrame with transaction data for one customer
            customer_id: Customer identifier
            date_col: Column name for transaction date
            amount_col: Column name for transaction amount
            category_col: Column name for product category (optional)
            is_promo_col: Column name for promo flag (optional)
            discount_col: Column name for discount amount (optional)

        Returns:
            Dictionary of behavioral features
        """
        if len(transactions) == 0:
            return {name: 0.0 for name in self.feature_names}

        # Ensure datetime
        transactions = transactions.copy()
        transactions[date_col] = pd.to_datetime(transactions[date_col])
        transactions = transactions.sort_values(date_col)

        features = {}

        # Time between visits
        if len(transactions) > 1:
            days_between = transactions[date_col].diff().dt.days.dropna()
            features["avg_days_between_visits"] = days_between.mean()
            features["std_days_between_visits"] = days_between.std() if len(days_between) > 1 else 0

            # Trend: are visits becoming more or less frequent?
            if len(days_between) > 2:
                half = len(days_between) // 2
                recent = days_between.iloc[-half:].mean()
                older = days_between.iloc[:half].mean()
                features["trend_days_between_visits"] = older - recent  # Positive = more frequent
            else:
                features["trend_days_between_visits"] = 0
        else:
            features["avg_days_between_visits"] = 0
            features["std_days_between_visits"] = 0
            features["trend_days_between_visits"] = 0

        # Basket size
        features["avg_basket_size"] = transactions[amount_col].mean()
        features["std_basket_size"] = transactions[amount_col].std() if len(transactions) > 1 else 0

        # Time preferences
        features["preferred_day_of_week"] = transactions[date_col].dt.dayofweek.mode().iloc[0] if len(transactions) > 0 else 0

        if hasattr(transactions[date_col].dt, 'hour'):
            hours = transactions[date_col].dt.hour
            features["preferred_hour"] = hours.mode().iloc[0] if len(hours) > 0 else 12
            features["morning_ratio"] = (hours < 12).mean()
            features["evening_ratio"] = (hours >= 18).mean()
        else:
            features["preferred_hour"] = 12
            features["morning_ratio"] = 0.5
            features["evening_ratio"] = 0.25

        features["weekend_ratio"] = (transactions[date_col].dt.dayofweek >= 5).mean()

        # Category diversity
        if category_col and category_col in transactions.columns:
            features["category_diversity"] = transactions[category_col].nunique()
        else:
            features["category_diversity"] = 0

        # Brand loyalty (placeholder - would need brand data)
        features["brand_loyalty"] = 0.0

        # Promo sensitivity
        if is_promo_col and is_promo_col in transactions.columns:
            features["promo_sensitivity"] = transactions[is_promo_col].mean()
        else:
            features["promo_sensitivity"] = 0.0

        # Tenure
        first_date = transactions[date_col].min()
        last_date = transactions[date_col].max()
        features["days_since_first_purchase"] = (datetime.now() - first_date).days

        # Purchase velocity (purchases per month)
        tenure_months = max(1, (last_date - first_date).days / 30)
        features["purchase_velocity"] = len(transactions) / tenure_months

        # Discount behavior
        if discount_col and discount_col in transactions.columns:
            total_amount = transactions[amount_col].sum()
            total_discount = transactions[discount_col].sum()
            features["avg_discount_rate"] = total_discount / total_amount if total_amount > 0 else 0
        else:
            features["avg_discount_rate"] = 0.0

        # Return rate (placeholder)
        features["return_rate"] = 0.0

        return features


class ExternalFactorEncoder:
    """Encode external factors: seasonality, holidays, weather, salary cycles."""

    # Russian holidays (approximate dates)
    HOLIDAYS = {
        (1, 1): "new_year",
        (1, 7): "christmas",
        (2, 23): "defender_day",
        (3, 8): "womens_day",
        (5, 1): "labor_day",
        (5, 9): "victory_day",
        (6, 12): "russia_day",
        (11, 4): "unity_day",
    }

    # Salary periods (typical Russian salary dates)
    SALARY_DAYS = [5, 10, 15, 20, 25]  # Common salary days

    def __init__(self):
        self.feature_names = [
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_holiday",
            "holiday_proximity",
            "is_salary_period",
            "days_to_salary",
            "season",
            # Cyclical encodings
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_of_month_sin",
            "day_of_month_cos",
        ]

    def encode(self, date: datetime) -> dict[str, float]:
        """Encode external factors for a given date."""

        features = {}

        # Basic date features
        features["day_of_week"] = date.weekday()
        features["day_of_month"] = date.day
        features["month"] = date.month
        features["quarter"] = (date.month - 1) // 3 + 1

        # Binary flags
        features["is_weekend"] = 1 if date.weekday() >= 5 else 0
        features["is_month_start"] = 1 if date.day <= 5 else 0
        features["is_month_end"] = 1 if date.day >= 25 else 0

        # Holiday detection
        is_holiday = (date.month, date.day) in self.HOLIDAYS
        features["is_holiday"] = 1 if is_holiday else 0

        # Days to nearest holiday
        min_distance = 365
        for (m, d), _ in self.HOLIDAYS.items():
            try:
                holiday_date = datetime(date.year, m, d)
                distance = abs((date - holiday_date).days)
                min_distance = min(min_distance, distance)
            except ValueError:
                continue
        features["holiday_proximity"] = max(0, 7 - min_distance) / 7  # 0-1, higher near holiday

        # Salary period
        is_salary = any(abs(date.day - sd) <= 2 for sd in self.SALARY_DAYS)
        features["is_salary_period"] = 1 if is_salary else 0

        days_to_salary = min(abs(date.day - sd) for sd in self.SALARY_DAYS)
        features["days_to_salary"] = days_to_salary

        # Season (0=winter, 1=spring, 2=summer, 3=fall)
        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        features["season"] = season_map[date.month]

        # Cyclical encodings (for neural networks)
        features["day_of_week_sin"] = np.sin(2 * np.pi * date.weekday() / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * date.weekday() / 7)
        features["month_sin"] = np.sin(2 * np.pi * (date.month - 1) / 12)
        features["month_cos"] = np.cos(2 * np.pi * (date.month - 1) / 12)
        features["day_of_month_sin"] = np.sin(2 * np.pi * (date.day - 1) / 31)
        features["day_of_month_cos"] = np.cos(2 * np.pi * (date.day - 1) / 31)

        return features

    def encode_batch(self, dates: list[datetime]) -> pd.DataFrame:
        """Encode external factors for multiple dates."""
        return pd.DataFrame([self.encode(d) for d in dates])


class CommunicationChannelEncoder:
    """Encode communication channel features."""

    CHANNELS = ["push", "email", "sms", "app", "web", "in_store"]

    def __init__(self):
        self.feature_names = [
            "channel_push",
            "channel_email",
            "channel_sms",
            "channel_app",
            "channel_web",
            "channel_in_store",
            "last_channel",
            "channel_diversity",
            "push_open_rate",
            "email_open_rate",
            "sms_response_rate",
        ]

    def encode(
        self,
        channel: str,
        channel_history: Optional[list[str]] = None,
        open_rates: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Encode channel features."""

        features = {}

        # One-hot encoding of current channel
        for ch in self.CHANNELS:
            features[f"channel_{ch}"] = 1 if channel == ch else 0

        # Last channel used (numerical encoding)
        features["last_channel"] = self.CHANNELS.index(channel) if channel in self.CHANNELS else -1

        # Channel diversity (how many channels customer used)
        if channel_history:
            features["channel_diversity"] = len(set(channel_history))
        else:
            features["channel_diversity"] = 1

        # Open/response rates
        if open_rates:
            features["push_open_rate"] = open_rates.get("push", 0.0)
            features["email_open_rate"] = open_rates.get("email", 0.0)
            features["sms_response_rate"] = open_rates.get("sms", 0.0)
        else:
            features["push_open_rate"] = 0.0
            features["email_open_rate"] = 0.0
            features["sms_response_rate"] = 0.0

        return features


def create_uplift_features(
    customer_data: pd.DataFrame,
    transactions: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    customer_id_col: str = "customer_id",
    date_col: str = "transaction_date",
    amount_col: str = "amount",
) -> pd.DataFrame:
    """
    Create comprehensive feature set for uplift modeling.

    Args:
        customer_data: Customer dimension table
        transactions: Transaction fact table
        reference_date: Date to calculate features from (default: now)
        customer_id_col: Column name for customer ID
        date_col: Column name for transaction date
        amount_col: Column name for transaction amount

    Returns:
        DataFrame with all features
    """
    if reference_date is None:
        reference_date = datetime.now()

    logger.info(
        "Creating uplift features",
        n_customers=len(customer_data),
        n_transactions=len(transactions),
    )

    # Initialize extractors
    behavioral_extractor = BehavioralFeatureExtractor()
    external_encoder = ExternalFactorEncoder()

    # Calculate RFM for all customers
    customer_rfm = []

    for cid in customer_data[customer_id_col].unique():
        cust_txns = transactions[transactions[customer_id_col] == cid]

        if len(cust_txns) == 0:
            continue

        cust_txns[date_col] = pd.to_datetime(cust_txns[date_col])

        recency = (reference_date - cust_txns[date_col].max()).days
        frequency = len(cust_txns)
        monetary = cust_txns[amount_col].sum()

        customer_rfm.append({
            customer_id_col: cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
        })

    rfm_df = pd.DataFrame(customer_rfm)

    # Fit RFM calculator and score
    rfm_calc = RFMCalculator()
    rfm_calc.fit(
        rfm_df["recency"].values,
        rfm_df["frequency"].values,
        rfm_df["monetary"].values,
    )

    rfm_features = rfm_calc.calculate_batch(
        rfm_df["recency"].values,
        rfm_df["frequency"].values,
        rfm_df["monetary"].values,
    )
    rfm_features[customer_id_col] = rfm_df[customer_id_col].values

    # Add external factors for reference date
    external_features = external_encoder.encode(reference_date)
    for key, value in external_features.items():
        rfm_features[f"ext_{key}"] = value

    # Merge with customer data
    result = customer_data.merge(rfm_features, on=customer_id_col, how="left")

    logger.info("Uplift features created", n_features=len(result.columns))

    return result

