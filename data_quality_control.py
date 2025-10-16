"""
Data Quality Control and Intelligent Sampling Module for Drug Discovery Toolkit

This module provides comprehensive data quality control, intelligent sampling strategies,
and validation mechanisms to prevent data quality degradation during model training.

Author: Drug Discovery Toolkit Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import logging
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import json
import hashlib
from datetime import datetime
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.ML.Cluster import Butina
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""

    total_samples: int = 0
    unique_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    diversity_score: float = 0.0
    balance_score: float = 0.0
    coverage_score: float = 0.0
    outlier_count: int = 0
    missing_value_ratio: float = 0.0
    chemical_diversity: Optional[float] = None
    sequence_diversity: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SamplingConfig:
    """Configuration for data sampling strategies"""

    strategy: str = "stratified"  # stratified, diversity, cluster, hybrid
    target_size: Optional[int] = None
    target_fraction: Optional[float] = None
    maintain_class_ratio: bool = True
    ensure_diversity: bool = True
    min_samples_per_class: int = 5
    diversity_threshold: float = 0.7
    random_seed: int = 42
    validation_split: float = 0.1
    test_split: float = 0.1
    k_fold: Optional[int] = None
    preserve_temporal_order: bool = False
    remove_outliers: bool = False
    outlier_threshold: float = 3.0


class DataQualityController:
    """
    Main class for data quality control and intelligent sampling
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        self.metrics_history = []
        self.sampling_log = []

    def analyze_data_quality(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        labels: Optional[Union[pd.Series, np.ndarray, List]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> DataQualityMetrics:
        """
        Comprehensive data quality analysis

        Args:
            data: Input data
            labels: Target labels (optional)
            feature_names: Names of features (optional)

        Returns:
            DataQualityMetrics object with comprehensive metrics
        """
        metrics = DataQualityMetrics()

        # Convert to DataFrame for easier analysis
        if isinstance(data, (list, np.ndarray)):
            data = pd.DataFrame(data, columns=feature_names)

        # Basic statistics
        metrics.total_samples = len(data)
        metrics.unique_samples = len(data.drop_duplicates())

        # Missing value analysis
        total_values = data.size
        missing_values = data.isnull().sum().sum()
        metrics.missing_value_ratio = (
            missing_values / total_values if total_values > 0 else 0
        )

        # Feature statistics
        for col in data.columns:
            if data[col].dtype in ["float64", "int64"]:
                metrics.feature_statistics[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "median": float(data[col].median()),
                    "missing_ratio": float(data[col].isnull().sum() / len(data)),
                }

        # Class distribution analysis
        if labels is not None:
            if isinstance(labels, pd.Series):
                labels = labels.values
            unique, counts = np.unique(labels, return_counts=True)
            metrics.class_distribution = dict(zip(map(str, unique), map(int, counts)))

            # Calculate balance score (0 = perfectly imbalanced, 1 = perfectly balanced)
            if len(counts) > 1:
                metrics.balance_score = 1 - (counts.std() / counts.mean())
            else:
                metrics.balance_score = 1.0

        # Diversity analysis
        metrics.diversity_score = self._calculate_diversity(data)

        # Outlier detection
        metrics.outlier_count = self._detect_outliers(data)

        # Coverage analysis
        metrics.coverage_score = self._calculate_coverage(data)

        logger.info(
            f"Data quality analysis complete: {metrics.total_samples} samples analyzed"
        )
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_diversity(self, data: pd.DataFrame) -> float:
        """Calculate diversity score based on feature variance"""
        if len(data) < 2:
            return 0.0

        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) == 0:
            return 0.0

        # Sample data if too large
        sample_size = min(1000, len(data))
        if len(data) > sample_size:
            data_sample = data.sample(
                n=sample_size, random_state=self.config.random_seed
            )
        else:
            data_sample = data

        # Calculate pairwise distances
        numeric_data = data_sample[numeric_cols].fillna(0)
        if len(numeric_data) > 1:
            distances = pdist(numeric_data, metric="euclidean")
            diversity = np.mean(distances) / (np.std(distances) + 1e-8)
            return min(1.0, diversity / 10)  # Normalize to [0, 1]
        return 0.0

    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """Detect outliers using IQR method"""
        outlier_count = 0
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_count += outliers

        return outlier_count

    def _calculate_coverage(self, data: pd.DataFrame) -> float:
        """Calculate feature space coverage"""
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        if len(numeric_cols) == 0:
            return 0.0

        coverage_scores = []
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                # Calculate range coverage
                data_range = col_data.max() - col_data.min()
                theoretical_range = 6 * col_data.std()  # 6-sigma range
                coverage = min(1.0, data_range / (theoretical_range + 1e-8))
                coverage_scores.append(coverage)

        return np.mean(coverage_scores) if coverage_scores else 0.0

    def intelligent_sample(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        labels: Optional[Union[pd.Series, np.ndarray, List]] = None,
        features: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Perform intelligent sampling based on configuration

        Args:
            data: Input data
            labels: Target labels
            features: Feature matrix for diversity calculation
            smiles: SMILES strings for chemical diversity

        Returns:
            Tuple of (sampled_data, sampled_labels, sampling_info)
        """
        sampling_info = {
            "strategy": self.config.strategy,
            "original_size": len(data),
            "timestamp": datetime.now().isoformat(),
        }

        # Determine target size
        if self.config.target_size:
            target_size = min(self.config.target_size, len(data))
        elif self.config.target_fraction:
            target_size = int(len(data) * self.config.target_fraction)
        else:
            target_size = len(data)

        sampling_info["target_size"] = target_size

        # Select sampling strategy
        if self.config.strategy == "stratified":
            sampled_idx = self._stratified_sampling(data, labels, target_size)
        elif self.config.strategy == "diversity":
            sampled_idx = self._diversity_sampling(data, features, target_size)
        elif self.config.strategy == "cluster":
            sampled_idx = self._cluster_sampling(data, features, target_size)
        elif self.config.strategy == "chemical_diversity" and smiles:
            sampled_idx = self._chemical_diversity_sampling(smiles, target_size)
        elif self.config.strategy == "hybrid":
            sampled_idx = self._hybrid_sampling(data, labels, features, target_size)
        else:
            # Default: random sampling
            sampled_idx = np.random.choice(len(data), target_size, replace=False)

        # Apply sampling
        if isinstance(data, pd.DataFrame):
            sampled_data = data.iloc[sampled_idx]
        else:
            sampled_data = [data[i] for i in sampled_idx]

        sampled_labels = None
        if labels is not None:
            if isinstance(labels, (pd.Series, np.ndarray)):
                sampled_labels = labels[sampled_idx]
            else:
                sampled_labels = [labels[i] for i in sampled_idx]

        # Calculate quality metrics for sampled data
        sampled_metrics = self.analyze_data_quality(sampled_data, sampled_labels)

        sampling_info.update(
            {
                "final_size": len(sampled_idx),
                "indices": sampled_idx.tolist()
                if isinstance(sampled_idx, np.ndarray)
                else sampled_idx,
                "diversity_score": sampled_metrics.diversity_score,
                "balance_score": sampled_metrics.balance_score,
                "coverage_score": sampled_metrics.coverage_score,
            }
        )

        # Log sampling operation
        self.sampling_log.append(sampling_info)
        logger.info(
            f"Sampling complete: {sampling_info['original_size']} → {sampling_info['final_size']}"
        )

        return sampled_data, sampled_labels, sampling_info

    def _stratified_sampling(
        self,
        data: Union[pd.DataFrame, List],
        labels: Union[pd.Series, np.ndarray, List],
        target_size: int,
    ) -> np.ndarray:
        """Stratified sampling maintaining class distribution"""
        if labels is None:
            return np.random.choice(len(data), target_size, replace=False)

        # Convert to numpy for easier handling
        if isinstance(labels, pd.Series):
            labels = labels.values
        elif isinstance(labels, list):
            labels = np.array(labels)

        unique_classes, class_counts = np.unique(labels, return_counts=True)

        # Calculate samples per class
        total_samples = len(labels)
        sampled_indices = []

        for cls, count in zip(unique_classes, class_counts):
            class_ratio = count / total_samples
            n_samples = max(
                self.config.min_samples_per_class, int(target_size * class_ratio)
            )
            n_samples = min(n_samples, count)  # Can't sample more than available

            class_indices = np.where(labels == cls)[0]
            sampled = np.random.choice(class_indices, n_samples, replace=False)
            sampled_indices.extend(sampled)

        return np.array(sampled_indices)

    def _diversity_sampling(
        self,
        data: Union[pd.DataFrame, List],
        features: Optional[np.ndarray],
        target_size: int,
    ) -> np.ndarray:
        """Maximum diversity sampling using greedy farthest point algorithm"""
        if features is None:
            # Try to extract numeric features from data
            if isinstance(data, pd.DataFrame):
                features = data.select_dtypes(include=["float64", "int64"]).values
            else:
                return np.random.choice(len(data), target_size, replace=False)

        if len(features) <= target_size:
            return np.arange(len(features))

        # Start with random point
        selected = [np.random.randint(len(features))]
        remaining = set(range(len(features))) - set(selected)

        # Greedy selection
        while len(selected) < target_size:
            # Find point farthest from all selected points
            max_min_dist = -1
            best_point = None

            for point in remaining:
                # Calculate minimum distance to selected points
                min_dist = float("inf")
                for sel_point in selected:
                    dist = np.linalg.norm(features[point] - features[sel_point])
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_point = point

            if best_point is not None:
                selected.append(best_point)
                remaining.remove(best_point)
            else:
                break

        return np.array(selected)

    def _cluster_sampling(
        self,
        data: Union[pd.DataFrame, List],
        features: Optional[np.ndarray],
        target_size: int,
    ) -> np.ndarray:
        """Cluster-based sampling for representative subset"""
        if features is None:
            if isinstance(data, pd.DataFrame):
                features = data.select_dtypes(include=["float64", "int64"]).values
            else:
                return np.random.choice(len(data), target_size, replace=False)

        if len(features) <= target_size:
            return np.arange(len(features))

        # Perform k-means clustering
        n_clusters = min(target_size, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed)
        cluster_labels = kmeans.fit_predict(features)

        selected_indices = []

        # Select representative from each cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select point closest to centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(features[cluster_indices] - centroid, axis=1)
                selected = cluster_indices[np.argmin(distances)]
                selected_indices.append(selected)

        # If we need more samples, add random samples from clusters
        while len(selected_indices) < target_size:
            cluster_id = np.random.randint(n_clusters)
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            available = set(cluster_indices) - set(selected_indices)
            if available:
                selected_indices.append(np.random.choice(list(available)))

        return np.array(selected_indices[:target_size])

    def _chemical_diversity_sampling(
        self, smiles: List[str], target_size: int
    ) -> np.ndarray:
        """Chemical diversity sampling using molecular fingerprints"""
        if len(smiles) <= target_size:
            return np.arange(len(smiles))

        # Calculate molecular fingerprints
        valid_indices = []
        fingerprints = []

        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fingerprints.append(fp)
                valid_indices.append(i)

        if not fingerprints:
            return np.random.choice(len(smiles), target_size, replace=False)

        # Calculate Tanimoto distance matrix
        n_mols = len(fingerprints)
        distance_matrix = np.zeros((n_mols, n_mols))

        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                similarity = DataStructs.TanimotoSimilarity(
                    fingerprints[i], fingerprints[j]
                )
                distance = 1 - similarity
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Use Butina clustering for diverse selection
        distance_threshold = 0.4  # Tanimoto distance threshold
        clusters = self._butina_cluster(distance_matrix, distance_threshold)

        selected = []

        # Select one representative from each cluster
        for cluster in clusters:
            if cluster:
                # Select medoid of cluster
                if len(cluster) == 1:
                    selected.append(valid_indices[cluster[0]])
                else:
                    cluster_distances = distance_matrix[np.ix_(cluster, cluster)]
                    medoid_idx = cluster[np.argmin(cluster_distances.sum(axis=1))]
                    selected.append(valid_indices[medoid_idx])

        # Add more samples if needed
        while len(selected) < target_size and len(selected) < len(valid_indices):
            remaining = set(valid_indices) - set(selected)
            if remaining:
                # Add most distant molecule from selected set
                max_min_dist = -1
                best_mol = None

                for mol_idx in remaining:
                    mol_pos = valid_indices.index(mol_idx)
                    min_dist = min(
                        distance_matrix[mol_pos, valid_indices.index(s)]
                        for s in selected
                    )
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_mol = mol_idx

                if best_mol is not None:
                    selected.append(best_mol)

        return np.array(selected[:target_size])

    def _butina_cluster(
        self, distance_matrix: np.ndarray, threshold: float
    ) -> List[List[int]]:
        """Perform Butina clustering"""
        n = len(distance_matrix)
        neighbors = defaultdict(list)

        # Find neighbors for each point
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] <= threshold:
                    neighbors[i].append(j)
                    neighbors[j].append(i)

        # Sort by number of neighbors (descending)
        sorted_indices = sorted(range(n), key=lambda x: len(neighbors[x]), reverse=True)

        clusters = []
        assigned = set()

        for idx in sorted_indices:
            if idx not in assigned:
                cluster = [idx] + [n for n in neighbors[idx] if n not in assigned]
                clusters.append(cluster)
                assigned.update(cluster)

        return clusters

    def _hybrid_sampling(
        self,
        data: Union[pd.DataFrame, List],
        labels: Optional[Union[pd.Series, np.ndarray, List]],
        features: Optional[np.ndarray],
        target_size: int,
    ) -> np.ndarray:
        """Hybrid sampling combining stratification and diversity"""
        # First apply stratified sampling to get 2x target size
        intermediate_size = min(len(data), target_size * 2)

        if labels is not None:
            stratified_indices = self._stratified_sampling(
                data, labels, intermediate_size
            )
        else:
            stratified_indices = np.arange(len(data))

        # Then apply diversity sampling on stratified subset
        if features is not None:
            stratified_features = features[stratified_indices]
            diversity_indices = self._diversity_sampling(
                data=None, features=stratified_features, target_size=target_size
            )
            final_indices = stratified_indices[diversity_indices]
        else:
            final_indices = np.random.choice(
                stratified_indices, target_size, replace=False
            )

        return final_indices

    def create_splits(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        labels: Optional[Union[pd.Series, np.ndarray, List]] = None,
        features: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Create train/validation/test splits with quality control

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing data and labels
        """
        n_samples = len(data)

        # Calculate split sizes
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size

        logger.info(
            f"Creating splits: train={train_size}, val={val_size}, test={test_size}"
        )

        # Create stratified splits if labels provided
        if labels is not None and self.config.maintain_class_ratio:
            # First split: train+val vs test
            indices = np.arange(n_samples)
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=labels,
                random_state=self.config.random_seed,
            )

            # Second split: train vs val
            train_val_labels = (
                labels[train_val_idx]
                if isinstance(labels, np.ndarray)
                else [labels[i] for i in train_val_idx]
            )
            relative_val_size = val_size / (train_size + val_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                stratify=train_val_labels,
                random_state=self.config.random_seed + 1,
            )
        else:
            # Random splits
            indices = np.random.permutation(n_samples)
            train_idx = indices[:train_size]
            val_idx = indices[train_size : train_size + val_size]
            test_idx = indices[train_size + val_size :]

        # Apply diversity sampling if configured
        if self.config.ensure_diversity and features is not None:
            train_idx = self._ensure_split_diversity(train_idx, features)
            val_idx = self._ensure_split_diversity(val_idx, features)
            test_idx = self._ensure_split_diversity(test_idx, features)

        # Create split data
        splits = {}
        for split_name, split_idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            if isinstance(data, pd.DataFrame):
                split_data = data.iloc[split_idx]
            elif isinstance(data, np.ndarray):
                split_data = data[split_idx]
            else:
                split_data = [data[i] for i in split_idx]

            split_labels = None
            if labels is not None:
                if isinstance(labels, (pd.Series, np.ndarray)):
                    split_labels = labels[split_idx]
                else:
                    split_labels = [labels[i] for i in split_idx]

            splits[split_name] = (split_data, split_labels)

            # Log quality metrics for each split
            metrics = self.analyze_data_quality(split_data, split_labels)
            logger.info(
                f"{split_name} split: diversity={metrics.diversity_score:.3f}, "
                f"balance={metrics.balance_score:.3f}, coverage={metrics.coverage_score:.3f}"
            )

        return splits

    def _ensure_split_diversity(
        self, indices: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Ensure split has sufficient diversity"""
        split_features = features[indices]
        diversity = self._calculate_feature_diversity(split_features)

        if diversity < self.config.diversity_threshold:
            logger.warning(
                f"Low diversity detected ({diversity:.3f}), applying diversity enhancement"
            )
            # Apply diversity sampling to enhance diversity
            enhanced_indices = self._diversity_sampling(
                data=None, features=split_features, target_size=len(indices)
            )
            return indices[enhanced_indices]

        return indices

    def _calculate_feature_diversity(self, features: np.ndarray) -> float:
        """Calculate diversity score for features"""
        if len(features) < 2:
            return 0.0

        # Sample if too large
        if len(features) > 1000:
            sample_idx = np.random.choice(len(features), 1000, replace=False)
            features = features[sample_idx]

        # Calculate pairwise distances
        distances = pdist(features, metric="euclidean")

        # Normalize by feature dimensions
        n_features = features.shape[1] if len(features.shape) > 1 else 1
        normalized_distances = distances / np.sqrt(n_features)

        # Calculate diversity as mean normalized distance
        diversity = np.mean(normalized_distances)

        return min(1.0, diversity)  # Cap at 1.0

    def validate_sampling(
        self,
        original_data: Union[pd.DataFrame, np.ndarray],
        sampled_data: Union[pd.DataFrame, np.ndarray],
        original_labels: Optional[Union[pd.Series, np.ndarray]] = None,
        sampled_labels: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Validate that sampling maintains data quality

        Returns:
            Dictionary with validation results and warnings
        """
        validation_results = {"passed": True, "warnings": [], "metrics": {}}

        # Analyze original and sampled data
        original_metrics = self.analyze_data_quality(original_data, original_labels)
        sampled_metrics = self.analyze_data_quality(sampled_data, sampled_labels)

        validation_results["metrics"] = {
            "original": original_metrics.__dict__,
            "sampled": sampled_metrics.__dict__,
        }

        # Check for diversity degradation
        diversity_loss = (
            original_metrics.diversity_score - sampled_metrics.diversity_score
        )
        if diversity_loss > 0.2:
            validation_results["warnings"].append(
                f"Significant diversity loss: {diversity_loss:.3f}"
            )
            validation_results["passed"] = False

        # Check class balance
        if original_labels is not None and sampled_labels is not None:
            balance_loss = abs(
                original_metrics.balance_score - sampled_metrics.balance_score
            )
            if balance_loss > 0.15:
                validation_results["warnings"].append(
                    f"Class balance altered: {balance_loss:.3f}"
                )
                validation_results["passed"] = False

            # Check for missing classes
            original_classes = set(original_metrics.class_distribution.keys())
            sampled_classes = set(sampled_metrics.class_distribution.keys())
            missing_classes = original_classes - sampled_classes

            if missing_classes:
                validation_results["warnings"].append(
                    f"Missing classes after sampling: {missing_classes}"
                )
                validation_results["passed"] = False

        # Check coverage
        coverage_loss = original_metrics.coverage_score - sampled_metrics.coverage_score
        if coverage_loss > 0.25:
            validation_results["warnings"].append(
                f"Feature space coverage reduced: {coverage_loss:.3f}"
            )
            validation_results["passed"] = False

        # Check sample uniqueness
        duplicate_ratio = 1 - (
            sampled_metrics.unique_samples / sampled_metrics.total_samples
        )
        if duplicate_ratio > 0.05:
            validation_results["warnings"].append(
                f"High duplicate ratio in sampled data: {duplicate_ratio:.3f}"
            )
            validation_results["passed"] = False

        # Log validation results
        if validation_results["passed"]:
            logger.info("✓ Sampling validation passed")
        else:
            logger.warning(
                f"✗ Sampling validation failed with {len(validation_results['warnings'])} warnings"
            )
            for warning in validation_results["warnings"]:
                logger.warning(f"  - {warning}")

        return validation_results

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data quality and sampling report

        Args:
            output_path: Optional path to save report as JSON

        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config.__dict__,
            "metrics_history": [m.__dict__ for m in self.metrics_history],
            "sampling_operations": self.sampling_log,
            "summary": {
                "total_operations": len(self.sampling_log),
                "total_samples_processed": sum(
                    m.total_samples for m in self.metrics_history
                ),
                "average_diversity": np.mean(
                    [m.diversity_score for m in self.metrics_history]
                ),
                "average_balance": np.mean(
                    [m.balance_score for m in self.metrics_history]
                ),
                "average_coverage": np.mean(
                    [m.coverage_score for m in self.metrics_history]
                ),
            },
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")

        return report


class TemporalSampler:
    """
    Specialized sampler for time-series and temporal data
    """

    def __init__(self, preserve_order: bool = True, window_size: Optional[int] = None):
        self.preserve_order = preserve_order
        self.window_size
