"""Causal Discovery algorithms for automatic causal graph construction.

This module implements algorithms for discovering causal relationships from
observational data:
- PC Algorithm (Peter-Clark): Constraint-based approach
- Correlation-based heuristics
- Domain knowledge integration

Note: For production use, consider integrating with specialized libraries:
- causal-learn (py-causal)
- NOTEARS (differentiable structure learning)
- DoWhy for validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Literal
from itertools import combinations

from meridian.core.logging import get_logger


logger = get_logger(__name__)


@dataclass
class CausalEdge:
    """Represents a causal edge in the graph."""

    source: str
    target: str
    edge_type: Literal["directed", "undirected", "bidirected"] = "directed"
    strength: float = 0.0  # Correlation or effect strength
    confidence: float = 0.0  # Confidence in the edge (1 - p_value)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "strength": self.strength,
            "confidence": self.confidence,
        }


@dataclass
class CausalGraph:
    """Causal graph representation."""

    nodes: list[str]
    edges: list[CausalEdge]
    adjacency_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
        }

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes of a given node."""
        return [e.source for e in self.edges if e.target == node]

    def get_children(self, node: str) -> list[str]:
        """Get child nodes of a given node."""
        return [e.target for e in self.edges if e.source == node]

    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists."""
        return any(
            e.source == source and e.target == target
            for e in self.edges
        )

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix."""
        n = len(self.nodes)
        node_idx = {name: i for i, name in enumerate(self.nodes)}

        adj = np.zeros((n, n))
        for edge in self.edges:
            i = node_idx.get(edge.source)
            j = node_idx.get(edge.target)
            if i is not None and j is not None:
                adj[i, j] = edge.strength if edge.strength != 0 else 1

        return adj


class ConditionalIndependenceTest:
    """Statistical tests for conditional independence."""

    @staticmethod
    def partial_correlation(
        data: pd.DataFrame,
        x: str,
        y: str,
        z: list[str],
    ) -> tuple[float, float]:
        """
        Compute partial correlation between x and y given z.

        Uses recursive formula for partial correlations.

        Args:
            data: DataFrame with all variables
            x: First variable
            y: Second variable
            z: Conditioning set

        Returns:
            Tuple of (partial_correlation, p_value)
        """
        if len(z) == 0:
            # Simple correlation
            corr, p_val = stats.pearsonr(data[x], data[y])
            return corr, p_val

        if len(z) == 1:
            # First-order partial correlation
            z_var = z[0]

            rxy, _ = stats.pearsonr(data[x], data[y])
            rxz, _ = stats.pearsonr(data[x], data[z_var])
            ryz, _ = stats.pearsonr(data[y], data[z_var])

            numerator = rxy - rxz * ryz
            denominator = np.sqrt((1 - rxz**2) * (1 - ryz**2))

            if denominator < 1e-10:
                return 0.0, 1.0

            partial_corr = numerator / denominator

        else:
            # Higher-order: use regression residuals
            from sklearn.linear_model import LinearRegression

            Z = data[z].values

            # Regress x on z
            reg_x = LinearRegression().fit(Z, data[x])
            residual_x = data[x] - reg_x.predict(Z)

            # Regress y on z
            reg_y = LinearRegression().fit(Z, data[y])
            residual_y = data[y] - reg_y.predict(Z)

            # Correlation of residuals
            partial_corr, _ = stats.pearsonr(residual_x, residual_y)

        # P-value using Fisher's z-transform
        n = len(data)
        dof = n - len(z) - 3

        if dof <= 0:
            return partial_corr, 1.0

        # Fisher's z-transform
        z_stat = 0.5 * np.log((1 + partial_corr + 1e-10) / (1 - partial_corr + 1e-10))
        z_stat = z_stat * np.sqrt(dof)

        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return partial_corr, p_value

    @staticmethod
    def g_squared_test(
        data: pd.DataFrame,
        x: str,
        y: str,
        z: list[str],
        n_bins: int = 5,
    ) -> tuple[float, float]:
        """
        G-squared test for conditional independence (for discrete/binned data).

        Args:
            data: DataFrame with all variables
            x: First variable
            y: Second variable
            z: Conditioning set
            n_bins: Number of bins for continuous variables

        Returns:
            Tuple of (g_squared_statistic, p_value)
        """
        # Discretize continuous variables
        def discretize(series):
            if series.dtype in ['object', 'category', 'bool']:
                return series
            return pd.cut(series, bins=n_bins, labels=False)

        x_disc = discretize(data[x])
        y_disc = discretize(data[y])

        if len(z) == 0:
            # Marginal independence
            contingency = pd.crosstab(x_disc, y_disc)
            g_stat, p_val, dof, expected = stats.chi2_contingency(contingency, lambda_="log-likelihood")
            return g_stat, p_val

        # Conditional independence: stratify by z
        z_disc = data[z].apply(discretize)

        # Group by z and sum G-squared
        g_total = 0
        dof_total = 0

        for _, group in data.groupby(list(z_disc.columns)):
            if len(group) < 5:
                continue

            try:
                contingency = pd.crosstab(
                    discretize(group[x]),
                    discretize(group[y])
                )
                if contingency.size > 1:
                    g_stat, _, dof, _ = stats.chi2_contingency(contingency, lambda_="log-likelihood")
                    g_total += g_stat
                    dof_total += dof
            except Exception:
                continue

        if dof_total == 0:
            return 0.0, 1.0

        p_value = 1 - stats.chi2.cdf(g_total, dof_total)

        return g_total, p_value


class PCAlgorithm:
    """
    PC Algorithm for causal structure learning.

    The PC algorithm works in two phases:
    1. Skeleton discovery: Find edges using conditional independence tests
    2. Edge orientation: Orient edges using v-structures and propagation rules

    References:
    - Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_set: int = 3,
        test: Literal["partial_corr", "g_squared"] = "partial_corr",
    ):
        """
        Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
            max_conditioning_set: Maximum size of conditioning set
            test: Type of independence test
        """
        self.alpha = alpha
        self.max_conditioning_set = max_conditioning_set
        self.test = test

        self._ci_test = ConditionalIndependenceTest()
        self._separation_sets: dict[tuple[str, str], list[str]] = {}

    def fit(
        self,
        data: pd.DataFrame,
        forbidden_edges: Optional[list[tuple[str, str]]] = None,
        required_edges: Optional[list[tuple[str, str]]] = None,
    ) -> CausalGraph:
        """
        Learn causal graph from data.

        Args:
            data: DataFrame with variables as columns
            forbidden_edges: Edges that cannot exist (domain knowledge)
            required_edges: Edges that must exist (domain knowledge)

        Returns:
            CausalGraph object
        """
        logger.info(
            "Running PC algorithm",
            n_samples=len(data),
            n_variables=len(data.columns),
            alpha=self.alpha,
        )

        nodes = list(data.columns)
        forbidden = set(forbidden_edges or [])
        required = set(required_edges or [])

        # Phase 1: Skeleton discovery
        skeleton = self._learn_skeleton(data, nodes, forbidden)

        # Add required edges
        for src, tgt in required:
            if src in nodes and tgt in nodes:
                if not any(e[0] == src and e[1] == tgt for e in skeleton):
                    skeleton.append((src, tgt, 0.0, 0.0))

        # Phase 2: Edge orientation
        edges = self._orient_edges(nodes, skeleton, data)

        graph = CausalGraph(nodes=nodes, edges=edges)

        logger.info(
            "PC algorithm complete",
            n_edges=len(edges),
        )

        return graph

    def _learn_skeleton(
        self,
        data: pd.DataFrame,
        nodes: list[str],
        forbidden: set[tuple[str, str]],
    ) -> list[tuple[str, str, float, float]]:
        """Learn the skeleton (undirected edges) of the graph."""

        # Start with complete graph
        edges = {}
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i < j:
                    if (x, y) not in forbidden and (y, x) not in forbidden:
                        edges[(x, y)] = True

        # Iteratively remove edges
        for cond_size in range(self.max_conditioning_set + 1):
            logger.debug(f"Testing conditioning sets of size {cond_size}")

            edges_to_remove = []

            for (x, y), exists in list(edges.items()):
                if not exists:
                    continue

                # Get adjacent nodes (potential conditioning sets)
                neighbors_x = [
                    n for (a, b), e in edges.items() if e
                    for n in ([b] if a == x else [a] if b == x else [])
                    if n != y
                ]

                # Test all conditioning sets of current size
                for z in combinations(neighbors_x, cond_size):
                    z_list = list(z)

                    if self.test == "partial_corr":
                        corr, p_val = self._ci_test.partial_correlation(data, x, y, z_list)
                    else:
                        corr, p_val = self._ci_test.g_squared_test(data, x, y, z_list)

                    if p_val > self.alpha:
                        # Conditionally independent - remove edge
                        edges_to_remove.append((x, y))
                        self._separation_sets[(x, y)] = z_list
                        self._separation_sets[(y, x)] = z_list
                        break

            for edge in edges_to_remove:
                edges[edge] = False

        # Calculate edge strengths
        result = []
        for (x, y), exists in edges.items():
            if exists:
                corr, p_val = self._ci_test.partial_correlation(data, x, y, [])
                result.append((x, y, abs(corr), 1 - p_val))

        return result

    def _orient_edges(
        self,
        nodes: list[str],
        skeleton: list[tuple[str, str, float, float]],
        data: pd.DataFrame,
    ) -> list[CausalEdge]:
        """Orient edges in the skeleton."""

        # Create adjacency list
        adjacent = {n: set() for n in nodes}
        edge_data = {}

        for x, y, strength, conf in skeleton:
            adjacent[x].add(y)
            adjacent[y].add(x)
            edge_data[(x, y)] = (strength, conf)
            edge_data[(y, x)] = (strength, conf)

        # Track directed edges: directed[x][y] = True means x -> y
        directed = {n: {} for n in nodes}

        # Rule 1: Orient v-structures (x -> z <- y where x and y not adjacent)
        for z in nodes:
            neighbors = list(adjacent[z])
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if y not in adjacent[x]:
                        # Check if z is in the separation set
                        sep_set = self._separation_sets.get((x, y), [])
                        if z not in sep_set:
                            # x -> z <- y (v-structure)
                            directed[x][z] = True
                            directed[y][z] = True

        # Rule 2-4: Meek's rules for edge orientation (simplified)
        changed = True
        while changed:
            changed = False

            for x in nodes:
                for y in adjacent[x]:
                    if y in directed[x] or x in directed[y]:
                        continue

                    # Rule 2: If x -> z -> y, then x -> y
                    for z in adjacent[x]:
                        if z in directed[x] and y in directed.get(z, {}):
                            directed[x][y] = True
                            changed = True
                            break

                    # Rule 3: If x -> z <- y and x - y, orient x -> y to avoid cycle
                    # (simplified version)

        # Build final edge list
        edges = []
        processed = set()

        for x, y, strength, conf in skeleton:
            if (x, y) in processed or (y, x) in processed:
                continue

            processed.add((x, y))

            if y in directed.get(x, {}):
                edges.append(CausalEdge(
                    source=x,
                    target=y,
                    edge_type="directed",
                    strength=strength,
                    confidence=conf,
                ))
            elif x in directed.get(y, {}):
                edges.append(CausalEdge(
                    source=y,
                    target=x,
                    edge_type="directed",
                    strength=strength,
                    confidence=conf,
                ))
            else:
                # Undirected - keep as is or use heuristics
                edges.append(CausalEdge(
                    source=x,
                    target=y,
                    edge_type="undirected",
                    strength=strength,
                    confidence=conf,
                ))

        return edges


class CausalDiscovery:
    """High-level interface for causal discovery."""

    def __init__(
        self,
        method: Literal["pc", "correlation"] = "pc",
        **kwargs,
    ):
        self.method = method
        self.kwargs = kwargs

        if method == "pc":
            self._algorithm = PCAlgorithm(**kwargs)
        else:
            self._algorithm = None

    def discover(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        forbidden_edges: Optional[list[tuple[str, str]]] = None,
        required_edges: Optional[list[tuple[str, str]]] = None,
    ) -> CausalGraph:
        """
        Discover causal graph from data.

        Args:
            data: DataFrame with variables
            target: Target variable (for focused discovery)
            forbidden_edges: Edges that cannot exist
            required_edges: Edges that must exist

        Returns:
            CausalGraph object
        """
        if self.method == "pc":
            return self._algorithm.fit(data, forbidden_edges, required_edges)

        elif self.method == "correlation":
            return self._correlation_based(data, target)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _correlation_based(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        threshold: float = 0.3,
    ) -> CausalGraph:
        """Simple correlation-based discovery (for comparison/baseline)."""

        nodes = list(data.columns)
        edges = []

        corr_matrix = data.corr()

        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                if i >= j:
                    continue

                corr = corr_matrix.loc[x, y]

                if abs(corr) >= threshold:
                    # Calculate p-value
                    n = len(data)
                    t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                    edges.append(CausalEdge(
                        source=x,
                        target=y,
                        edge_type="undirected",
                        strength=abs(corr),
                        confidence=1 - p_val,
                    ))

        return CausalGraph(nodes=nodes, edges=edges)

    def validate_with_domain_knowledge(
        self,
        graph: CausalGraph,
        expected_causes: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Validate discovered graph against domain knowledge.

        Args:
            graph: Discovered causal graph
            expected_causes: Dict mapping effects to their expected causes

        Returns:
            Validation metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for effect, expected in expected_causes.items():
            discovered_parents = graph.get_parents(effect)

            for cause in expected:
                if cause in discovered_parents:
                    true_positives += 1
                else:
                    false_negatives += 1

            for parent in discovered_parents:
                if parent not in expected:
                    false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

