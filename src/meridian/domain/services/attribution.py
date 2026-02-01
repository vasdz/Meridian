"""Multi-touch Attribution using Shapley Values and Markov Chains.

This module implements attribution models for understanding channel contributions:
- Shapley Value Attribution: Game-theoretic fair attribution
- Markov Chain Attribution: Transition probability-based attribution
- Last-touch / First-touch baselines

Critical for retail contact policy optimization.
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Callable

from meridian.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class AttributionResult:
    """Result of attribution calculation."""

    channel: str
    attribution_value: float  # Contribution (0-1 scale relative to total)
    absolute_value: float  # Actual value attributed
    conversion_probability: float  # P(conversion | channel in path)
    avg_position: float  # Average position in conversion paths
    frequency: int  # How often channel appears

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "attribution_value": self.attribution_value,
            "absolute_value": self.absolute_value,
            "conversion_probability": self.conversion_probability,
            "avg_position": self.avg_position,
            "frequency": self.frequency,
        }


@dataclass
class AttributionReport:
    """Complete attribution report."""

    model_name: str
    total_conversions: int
    total_value: float
    channel_attributions: list[AttributionResult]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_conversions": self.total_conversions,
            "total_value": self.total_value,
            "channels": [attr.to_dict() for attr in self.channel_attributions],
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([attr.to_dict() for attr in self.channel_attributions])


class ShapleyAttribution:
    """
    Shapley Value based multi-touch attribution.

    Shapley values provide a fair way to distribute credit among channels
    based on their marginal contributions across all possible orderings.

    For n channels, this requires computing 2^n coalitions, which can be
    expensive. We provide exact computation for small n and sampling for larger.
    """

    def __init__(
        self,
        max_exact_channels: int = 10,
        n_samples: int = 10000,
    ):
        """
        Initialize Shapley attribution.

        Args:
            max_exact_channels: Use exact computation if n_channels <= this
            n_samples: Number of samples for approximate Shapley
        """
        self.max_exact_channels = max_exact_channels
        self.n_samples = n_samples

    def calculate(
        self,
        conversion_paths: list[list[str]],
        conversion_values: Optional[list[float]] = None,
        characteristic_function: Optional[Callable[[set], float]] = None,
    ) -> AttributionReport:
        """
        Calculate Shapley value attribution.

        Args:
            conversion_paths: List of paths, each path is a list of channels
            conversion_values: Value of each conversion (default: 1.0 each)
            characteristic_function: Custom value function v(S) for coalition S

        Returns:
            AttributionReport with Shapley values for each channel
        """
        if conversion_values is None:
            conversion_values = [1.0] * len(conversion_paths)

        total_value = sum(conversion_values)
        total_conversions = len(conversion_paths)

        # Get all unique channels
        all_channels = set()
        for path in conversion_paths:
            all_channels.update(path)
        channels = sorted(list(all_channels))

        logger.info(
            "Calculating Shapley attribution",
            n_conversions=total_conversions,
            n_channels=len(channels),
        )

        # Build characteristic function if not provided
        if characteristic_function is None:
            characteristic_function = self._build_characteristic_function(
                conversion_paths, conversion_values
            )

        # Calculate Shapley values
        n = len(channels)

        if n <= self.max_exact_channels:
            shapley_values = self._exact_shapley(channels, characteristic_function)
        else:
            shapley_values = self._approximate_shapley(channels, characteristic_function)

        # Build results
        results = []
        channel_stats = self._calculate_channel_stats(conversion_paths, conversion_values)

        for channel in channels:
            stats = channel_stats.get(channel, {})

            results.append(AttributionResult(
                channel=channel,
                attribution_value=shapley_values[channel] / total_value if total_value > 0 else 0,
                absolute_value=shapley_values[channel],
                conversion_probability=stats.get("conversion_prob", 0),
                avg_position=stats.get("avg_position", 0),
                frequency=stats.get("frequency", 0),
            ))

        # Sort by attribution value
        results.sort(key=lambda x: x.attribution_value, reverse=True)

        return AttributionReport(
            model_name="shapley",
            total_conversions=total_conversions,
            total_value=total_value,
            channel_attributions=results,
        )

    def _build_characteristic_function(
        self,
        conversion_paths: list[list[str]],
        conversion_values: list[float],
    ) -> Callable[[set], float]:
        """Build v(S) = total value of conversions where all channels in S are present."""

        path_sets = [set(path) for path in conversion_paths]

        def v(coalition: set) -> float:
            if not coalition:
                return 0.0

            total = 0.0
            for path_set, value in zip(path_sets, conversion_values):
                # Conversion happens if coalition covers the path
                if coalition.intersection(path_set):
                    total += value

            return total

        return v

    def _exact_shapley(
        self,
        channels: list[str],
        v: Callable[[set], float],
    ) -> dict[str, float]:
        """Compute exact Shapley values for all channels."""

        n = len(channels)
        shapley = {c: 0.0 for c in channels}

        # Factorial lookup
        factorials = [1]
        for i in range(1, n + 1):
            factorials.append(factorials[-1] * i)

        # For each channel, sum marginal contributions over all coalitions
        for i, channel in enumerate(channels):
            other_channels = [c for c in channels if c != channel]

            for size in range(n):
                for subset in combinations(other_channels, size):
                    S = set(subset)
                    S_with_i = S | {channel}

                    marginal = v(S_with_i) - v(S)

                    # Weight: |S|!(n-|S|-1)! / n!
                    weight = (factorials[size] * factorials[n - size - 1]) / factorials[n]

                    shapley[channel] += weight * marginal

        return shapley

    def _approximate_shapley(
        self,
        channels: list[str],
        v: Callable[[set], float],
    ) -> dict[str, float]:
        """Approximate Shapley values using Monte Carlo sampling."""

        shapley = {c: 0.0 for c in channels}
        n = len(channels)

        for _ in range(self.n_samples):
            # Random permutation
            perm = np.random.permutation(channels)

            coalition = set()
            prev_value = 0.0

            for channel in perm:
                coalition.add(channel)
                curr_value = v(coalition)

                # Marginal contribution
                shapley[channel] += curr_value - prev_value
                prev_value = curr_value

        # Average
        for channel in channels:
            shapley[channel] /= self.n_samples

        return shapley

    def _calculate_channel_stats(
        self,
        conversion_paths: list[list[str]],
        conversion_values: list[float],
    ) -> dict[str, dict]:
        """Calculate descriptive statistics for each channel."""

        stats = defaultdict(lambda: {
            "frequency": 0,
            "total_value": 0,
            "positions": [],
        })

        for path, value in zip(conversion_paths, conversion_values):
            for pos, channel in enumerate(path):
                stats[channel]["frequency"] += 1
                stats[channel]["total_value"] += value
                stats[channel]["positions"].append(pos / len(path) if len(path) > 0 else 0)

        result = {}
        total_conversions = len(conversion_paths)

        for channel, data in stats.items():
            result[channel] = {
                "frequency": data["frequency"],
                "conversion_prob": data["frequency"] / total_conversions if total_conversions > 0 else 0,
                "avg_position": np.mean(data["positions"]) if data["positions"] else 0,
            }

        return result


class MarkovAttribution:
    """
    Markov Chain based multi-touch attribution.

    Models customer journeys as a Markov chain where states are channels
    and transitions are observed movements between channels.

    Attribution is based on "removal effect": how much conversion probability
    drops when a channel is removed from the chain.
    """

    def __init__(self):
        self._transition_matrix: Optional[np.ndarray] = None
        self._states: Optional[list[str]] = None
        self._state_idx: Optional[dict[str, int]] = None

    def calculate(
        self,
        conversion_paths: list[list[str]],
        non_conversion_paths: Optional[list[list[str]]] = None,
        conversion_values: Optional[list[float]] = None,
    ) -> AttributionReport:
        """
        Calculate Markov chain attribution.

        Args:
            conversion_paths: Paths that led to conversion
            non_conversion_paths: Paths that didn't convert (optional but recommended)
            conversion_values: Value of each conversion

        Returns:
            AttributionReport with removal effect attributions
        """
        if conversion_values is None:
            conversion_values = [1.0] * len(conversion_paths)

        total_value = sum(conversion_values)
        total_conversions = len(conversion_paths)

        # Get all channels
        all_channels = set()
        for path in conversion_paths:
            all_channels.update(path)
        if non_conversion_paths:
            for path in non_conversion_paths:
                all_channels.update(path)

        # Add special states
        states = ["(start)"] + sorted(list(all_channels)) + ["(conversion)", "(null)"]
        self._states = states
        self._state_idx = {s: i for i, s in enumerate(states)}

        logger.info(
            "Calculating Markov attribution",
            n_conversions=total_conversions,
            n_channels=len(all_channels),
        )

        # Build transition matrix
        self._build_transition_matrix(conversion_paths, non_conversion_paths or [])

        # Calculate baseline conversion probability
        baseline_prob = self._calculate_conversion_probability()

        # Calculate removal effects
        removal_effects = {}
        channels = sorted(list(all_channels))

        for channel in channels:
            prob_without = self._calculate_conversion_probability(removed_channel=channel)
            removal_effect = baseline_prob - prob_without
            removal_effects[channel] = max(0, removal_effect)

        # Normalize to sum to 1
        total_effect = sum(removal_effects.values())

        # Build results
        results = []
        channel_stats = self._calculate_channel_stats(conversion_paths, conversion_values)

        for channel in channels:
            stats = channel_stats.get(channel, {})
            normalized = removal_effects[channel] / total_effect if total_effect > 0 else 0

            results.append(AttributionResult(
                channel=channel,
                attribution_value=normalized,
                absolute_value=normalized * total_value,
                conversion_probability=stats.get("conversion_prob", 0),
                avg_position=stats.get("avg_position", 0),
                frequency=stats.get("frequency", 0),
            ))

        results.sort(key=lambda x: x.attribution_value, reverse=True)

        return AttributionReport(
            model_name="markov",
            total_conversions=total_conversions,
            total_value=total_value,
            channel_attributions=results,
        )

    def _build_transition_matrix(
        self,
        conversion_paths: list[list[str]],
        non_conversion_paths: list[list[str]],
    ) -> None:
        """Build transition probability matrix."""

        n_states = len(self._states)
        counts = np.zeros((n_states, n_states))

        start_idx = self._state_idx["(start)"]
        conv_idx = self._state_idx["(conversion)"]
        null_idx = self._state_idx["(null)"]

        # Count transitions in conversion paths
        for path in conversion_paths:
            if not path:
                continue

            # Start -> first channel
            counts[start_idx, self._state_idx[path[0]]] += 1

            # Channel transitions
            for i in range(len(path) - 1):
                from_idx = self._state_idx[path[i]]
                to_idx = self._state_idx[path[i + 1]]
                counts[from_idx, to_idx] += 1

            # Last channel -> conversion
            counts[self._state_idx[path[-1]], conv_idx] += 1

        # Count transitions in non-conversion paths
        for path in non_conversion_paths:
            if not path:
                continue

            counts[start_idx, self._state_idx[path[0]]] += 1

            for i in range(len(path) - 1):
                from_idx = self._state_idx[path[i]]
                to_idx = self._state_idx[path[i + 1]]
                counts[from_idx, to_idx] += 1

            counts[self._state_idx[path[-1]], null_idx] += 1

        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        self._transition_matrix = counts / row_sums

        # Make absorbing states
        self._transition_matrix[conv_idx, :] = 0
        self._transition_matrix[conv_idx, conv_idx] = 1
        self._transition_matrix[null_idx, :] = 0
        self._transition_matrix[null_idx, null_idx] = 1

    def _calculate_conversion_probability(
        self,
        removed_channel: Optional[str] = None,
        max_steps: int = 100,
    ) -> float:
        """Calculate probability of reaching conversion from start."""

        if self._transition_matrix is None:
            return 0.0

        P = self._transition_matrix.copy()

        # Remove channel by redirecting to null
        if removed_channel and removed_channel in self._state_idx:
            channel_idx = self._state_idx[removed_channel]
            null_idx = self._state_idx["(null)"]

            # Redirect all transitions to this channel to null
            incoming = P[:, channel_idx].copy()
            P[:, channel_idx] = 0
            P[:, null_idx] += incoming

        # Absorbing Markov chain analysis
        # Find transient and absorbing states
        n = len(self._states)
        conv_idx = self._state_idx["(conversion)"]
        null_idx = self._state_idx["(null)"]
        start_idx = self._state_idx["(start)"]

        absorbing = {conv_idx, null_idx}
        transient = [i for i in range(n) if i not in absorbing]

        if not transient:
            return 0.0

        # Q = transitions among transient states
        # R = transitions from transient to absorbing
        Q = P[np.ix_(transient, transient)]
        R = P[np.ix_(transient, list(absorbing))]

        # Fundamental matrix N = (I - Q)^-1
        try:
            I = np.eye(len(transient))
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            # Use iterative approach if singular
            N = np.eye(len(transient))
            Q_power = np.eye(len(transient))
            for _ in range(max_steps):
                Q_power = Q_power @ Q
                N += Q_power
                if np.max(Q_power) < 1e-10:
                    break

        # Absorption probabilities B = N * R
        B = N @ R

        # Find start state in transient list
        if start_idx in transient:
            start_transient_idx = transient.index(start_idx)
            # Probability of reaching conversion (first absorbing state)
            conv_absorbing_idx = list(absorbing).index(conv_idx)
            return float(B[start_transient_idx, conv_absorbing_idx])

        return 0.0

    def _calculate_channel_stats(
        self,
        conversion_paths: list[list[str]],
        conversion_values: list[float],
    ) -> dict[str, dict]:
        """Calculate descriptive statistics for each channel."""

        stats = defaultdict(lambda: {"frequency": 0, "positions": []})

        for path, value in zip(conversion_paths, conversion_values):
            for pos, channel in enumerate(path):
                stats[channel]["frequency"] += 1
                stats[channel]["positions"].append(pos / len(path) if len(path) > 0 else 0)

        result = {}
        total_conversions = len(conversion_paths)

        for channel, data in stats.items():
            result[channel] = {
                "frequency": data["frequency"],
                "conversion_prob": data["frequency"] / total_conversions if total_conversions > 0 else 0,
                "avg_position": np.mean(data["positions"]) if data["positions"] else 0,
            }

        return result


class SimpleAttribution:
    """Simple attribution models for baseline comparison."""

    @staticmethod
    def last_touch(
        conversion_paths: list[list[str]],
        conversion_values: Optional[list[float]] = None,
    ) -> AttributionReport:
        """100% credit to last touchpoint."""

        if conversion_values is None:
            conversion_values = [1.0] * len(conversion_paths)

        attributions = defaultdict(float)
        frequencies = defaultdict(int)

        for path, value in zip(conversion_paths, conversion_values):
            if path:
                last_channel = path[-1]
                attributions[last_channel] += value
                frequencies[last_channel] += 1

        total_value = sum(conversion_values)

        results = [
            AttributionResult(
                channel=channel,
                attribution_value=value / total_value if total_value > 0 else 0,
                absolute_value=value,
                conversion_probability=frequencies[channel] / len(conversion_paths) if conversion_paths else 0,
                avg_position=1.0,  # Always last
                frequency=frequencies[channel],
            )
            for channel, value in attributions.items()
        ]

        results.sort(key=lambda x: x.attribution_value, reverse=True)

        return AttributionReport(
            model_name="last_touch",
            total_conversions=len(conversion_paths),
            total_value=total_value,
            channel_attributions=results,
        )

    @staticmethod
    def first_touch(
        conversion_paths: list[list[str]],
        conversion_values: Optional[list[float]] = None,
    ) -> AttributionReport:
        """100% credit to first touchpoint."""

        if conversion_values is None:
            conversion_values = [1.0] * len(conversion_paths)

        attributions = defaultdict(float)
        frequencies = defaultdict(int)

        for path, value in zip(conversion_paths, conversion_values):
            if path:
                first_channel = path[0]
                attributions[first_channel] += value
                frequencies[first_channel] += 1

        total_value = sum(conversion_values)

        results = [
            AttributionResult(
                channel=channel,
                attribution_value=value / total_value if total_value > 0 else 0,
                absolute_value=value,
                conversion_probability=frequencies[channel] / len(conversion_paths) if conversion_paths else 0,
                avg_position=0.0,  # Always first
                frequency=frequencies[channel],
            )
            for channel, value in attributions.items()
        ]

        results.sort(key=lambda x: x.attribution_value, reverse=True)

        return AttributionReport(
            model_name="first_touch",
            total_conversions=len(conversion_paths),
            total_value=total_value,
            channel_attributions=results,
        )

    @staticmethod
    def linear(
        conversion_paths: list[list[str]],
        conversion_values: Optional[list[float]] = None,
    ) -> AttributionReport:
        """Equal credit to all touchpoints."""

        if conversion_values is None:
            conversion_values = [1.0] * len(conversion_paths)

        attributions = defaultdict(float)
        frequencies = defaultdict(int)
        positions = defaultdict(list)

        for path, value in zip(conversion_paths, conversion_values):
            if path:
                credit = value / len(path)
                for pos, channel in enumerate(path):
                    attributions[channel] += credit
                    frequencies[channel] += 1
                    positions[channel].append(pos / len(path))

        total_value = sum(conversion_values)

        results = [
            AttributionResult(
                channel=channel,
                attribution_value=value / total_value if total_value > 0 else 0,
                absolute_value=value,
                conversion_probability=frequencies[channel] / len(conversion_paths) if conversion_paths else 0,
                avg_position=np.mean(positions[channel]) if positions[channel] else 0,
                frequency=frequencies[channel],
            )
            for channel, value in attributions.items()
        ]

        results.sort(key=lambda x: x.attribution_value, reverse=True)

        return AttributionReport(
            model_name="linear",
            total_conversions=len(conversion_paths),
            total_value=total_value,
            channel_attributions=results,
        )


def compare_attribution_models(
    conversion_paths: list[list[str]],
    conversion_values: Optional[list[float]] = None,
    non_conversion_paths: Optional[list[list[str]]] = None,
) -> pd.DataFrame:
    """
    Compare different attribution models.

    Args:
        conversion_paths: Paths that led to conversion
        conversion_values: Value of each conversion
        non_conversion_paths: Paths that didn't convert

    Returns:
        DataFrame comparing all models
    """
    logger.info("Comparing attribution models", n_paths=len(conversion_paths))

    results = {}

    # Simple models
    results["last_touch"] = SimpleAttribution.last_touch(conversion_paths, conversion_values)
    results["first_touch"] = SimpleAttribution.first_touch(conversion_paths, conversion_values)
    results["linear"] = SimpleAttribution.linear(conversion_paths, conversion_values)

    # Shapley
    shapley = ShapleyAttribution()
    results["shapley"] = shapley.calculate(conversion_paths, conversion_values)

    # Markov (if we have non-conversion paths)
    if non_conversion_paths:
        markov = MarkovAttribution()
        results["markov"] = markov.calculate(conversion_paths, non_conversion_paths, conversion_values)

    # Build comparison DataFrame
    all_channels = set()
    for report in results.values():
        for attr in report.channel_attributions:
            all_channels.add(attr.channel)

    comparison_data = []
    for channel in sorted(all_channels):
        row = {"channel": channel}
        for model_name, report in results.items():
            for attr in report.channel_attributions:
                if attr.channel == channel:
                    row[f"{model_name}_attribution"] = attr.attribution_value
                    break
            else:
                row[f"{model_name}_attribution"] = 0.0
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)

