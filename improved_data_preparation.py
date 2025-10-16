"""
Improved Data Preparation with Quality Control for Drug Discovery Toolkit

This module replaces arbitrary data reduction with intelligent sampling strategies
that maintain data quality, diversity, and representativeness.

Author: Drug Discovery Toolkit Team
Date: 2024
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter
import hashlib
import warnings

# Import RDKit for chemical analysis
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.ML.Cluster import Butina

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical diversity features will be limited.")

# Import sklearn for ML-based sampling
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImprovedDataPreparation:
    """
    Comprehensive data preparation with quality control and intelligent sampling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preparation with configuration

        Args:
            config: Configuration dictionary with following keys:
                - min_quality_score: Minimum acceptable quality score (0-1)
                - sampling_strategy: 'stratified', 'diversity', 'cluster', 'hybrid'
                - preserve_rare_samples: Whether to preserve rare/important samples
                - diversity_threshold: Minimum diversity score to maintain
                - validation_split: Fraction for validation (default 0.1)
                - test_split: Fraction for test (default 0.1)
                - random_seed: Random seed for reproducibility
        """
        self.config = config or self._get_default_config()
        self.quality_metrics = []
        self.sampling_history = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "min_quality_score": 0.7,
            "sampling_strategy": "hybrid",
            "preserve_rare_samples": True,
            "diversity_threshold": 0.6,
            "validation_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,
            "remove_duplicates": True,
            "remove_outliers": False,
            "outlier_threshold": 3.0,
            "min_samples_per_class": 5,
            "chemical_diversity_weight": 0.5,
            "structural_diversity_weight": 0.5,
        }

    def prepare_drug_protein_dataset(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Prepare drug-protein interaction dataset with quality control

        Args:
            drugs: List of drug entries with SMILES, properties, etc.
            proteins: List of protein entries with sequences, structures, etc.
            interactions: List of drug-protein interactions with affinities
            output_dir: Directory to save prepared data

        Returns:
            Dictionary with prepared datasets and quality metrics
        """
        logger.info(
            f"Starting data preparation for {len(drugs)} drugs, {len(proteins)} proteins"
        )

        # Step 1: Data cleaning and validation
        clean_drugs = self._clean_drug_data(drugs)
        clean_proteins = self._clean_protein_data(proteins)
        clean_interactions = self._clean_interaction_data(
            interactions, clean_drugs, clean_proteins
        )

        # Step 2: Calculate quality metrics
        quality_report = self._calculate_quality_metrics(
            clean_drugs, clean_proteins, clean_interactions
        )

        # Step 3: Check if quality meets threshold
        if quality_report["overall_score"] < self.config["min_quality_score"]:
            logger.warning(
                f"Data quality ({quality_report['overall_score']:.2f}) below threshold"
            )
            # Apply quality improvement strategies
            clean_drugs, clean_proteins, clean_interactions = (
                self._improve_data_quality(
                    clean_drugs, clean_proteins, clean_interactions
                )
            )
            # Recalculate metrics
            quality_report = self._calculate_quality_metrics(
                clean_drugs, clean_proteins, clean_interactions
            )

        # Step 4: Intelligent sampling instead of arbitrary reduction
        sampled_data = self._intelligent_sampling(
            clean_drugs, clean_proteins, clean_interactions
        )

        # Step 5: Create train/val/test splits with proper stratification
        splits = self._create_quality_controlled_splits(sampled_data)

        # Step 6: Save prepared data
        self._save_prepared_data(splits, output_dir)

        # Step 7: Generate comprehensive report
        report = self._generate_preparation_report(
            original_sizes={
                "drugs": len(drugs),
                "proteins": len(proteins),
                "interactions": len(interactions),
            },
            cleaned_sizes={
                "drugs": len(clean_drugs),
                "proteins": len(clean_proteins),
                "interactions": len(clean_interactions),
            },
            quality_report=quality_report,
            splits=splits,
            output_dir=output_dir,
        )

        return report

    def _clean_drug_data(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate drug data"""
        clean_drugs = []
        invalid_count = 0

        for drug in tqdm(drugs, desc="Cleaning drug data"):
            # Validate SMILES
            if "smiles" not in drug or not drug["smiles"]:
                invalid_count += 1
                continue

            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(drug["smiles"])
                if mol is None:
                    invalid_count += 1
                    continue
                # Standardize SMILES
                drug["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
                # Calculate molecular properties
                drug["molecular_weight"] = Descriptors.ExactMolWt(mol)
                drug["logp"] = Descriptors.MolLogP(mol)
                drug["num_rings"] = Chem.rdMolDescriptors.CalcNumRings(mol)

            # Remove duplicates based on canonical SMILES
            if self.config["remove_duplicates"]:
                smiles_key = drug.get("canonical_smiles", drug["smiles"])
                if not any(
                    d.get("canonical_smiles", d["smiles"]) == smiles_key
                    for d in clean_drugs
                ):
                    clean_drugs.append(drug)
            else:
                clean_drugs.append(drug)

        logger.info(
            f"Cleaned drugs: {len(clean_drugs)}/{len(drugs)} valid ({invalid_count} invalid)"
        )
        return clean_drugs

    def _clean_protein_data(
        self, proteins: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean and validate protein data"""
        clean_proteins = []
        invalid_count = 0

        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

        for protein in tqdm(proteins, desc="Cleaning protein data"):
            # Validate sequence
            if "sequence" not in protein or not protein["sequence"]:
                invalid_count += 1
                continue

            # Clean sequence
            sequence = protein["sequence"].upper().replace(" ", "").replace("\n", "")

            # Check for invalid amino acids
            if not all(aa in valid_amino_acids for aa in sequence):
                # Remove invalid characters
                sequence = "".join(aa for aa in sequence if aa in valid_amino_acids)
                if len(sequence) < 10:  # Too short after cleaning
                    invalid_count += 1
                    continue

            protein["clean_sequence"] = sequence
            protein["sequence_length"] = len(sequence)

            # Calculate sequence properties
            protein["sequence_complexity"] = (
                len(set(sequence)) / 20
            )  # Amino acid diversity

            clean_proteins.append(protein)

        logger.info(
            f"Cleaned proteins: {len(clean_proteins)}/{len(proteins)} valid ({invalid_count} invalid)"
        )
        return clean_proteins

    def _clean_interaction_data(
        self,
        interactions: List[Dict[str, Any]],
        clean_drugs: List[Dict[str, Any]],
        clean_proteins: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Clean and validate interaction data"""
        clean_interactions = []
        invalid_count = 0

        # Create lookup dictionaries
        drug_ids = {d["id"] for d in clean_drugs if "id" in d}
        protein_ids = {p["id"] for p in clean_proteins if "id" in p}

        for interaction in tqdm(interactions, desc="Cleaning interaction data"):
            # Check if both drug and protein exist in cleaned data
            if interaction.get("drug_id") not in drug_ids:
                invalid_count += 1
                continue
            if interaction.get("protein_id") not in protein_ids:
                invalid_count += 1
                continue

            # Validate affinity value
            if "affinity" in interaction:
                try:
                    affinity = float(interaction["affinity"])
                    if np.isnan(affinity) or np.isinf(affinity):
                        invalid_count += 1
                        continue
                    interaction["affinity"] = affinity
                except (ValueError, TypeError):
                    invalid_count += 1
                    continue

            clean_interactions.append(interaction)

        logger.info(
            f"Cleaned interactions: {len(clean_interactions)}/{len(interactions)} valid ({invalid_count} invalid)"
        )
        return clean_interactions

    def _calculate_quality_metrics(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "drug_metrics": self._calculate_drug_metrics(drugs),
            "protein_metrics": self._calculate_protein_metrics(proteins),
            "interaction_metrics": self._calculate_interaction_metrics(interactions),
        }

        # Calculate overall quality score
        scores = []
        if metrics["drug_metrics"]["diversity_score"] is not None:
            scores.append(metrics["drug_metrics"]["diversity_score"])
        if metrics["protein_metrics"]["diversity_score"] is not None:
            scores.append(metrics["protein_metrics"]["diversity_score"])
        scores.append(metrics["interaction_metrics"]["balance_score"])
        scores.append(metrics["interaction_metrics"]["coverage_score"])

        metrics["overall_score"] = np.mean(scores) if scores else 0.0

        return metrics

    def _calculate_drug_metrics(self, drugs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate drug-specific quality metrics"""
        metrics = {
            "count": len(drugs),
            "unique_count": len(
                set(d.get("canonical_smiles", d["smiles"]) for d in drugs)
            ),
            "diversity_score": None,
            "property_distribution": {},
        }

        if RDKIT_AVAILABLE and len(drugs) > 1:
            # Calculate chemical diversity
            fingerprints = []
            for drug in drugs[: min(1000, len(drugs))]:  # Sample for efficiency
                mol = Chem.MolFromSmiles(drug.get("canonical_smiles", drug["smiles"]))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                    fingerprints.append(fp)

            if len(fingerprints) > 1:
                # Calculate average Tanimoto distance
                similarities = []
                for i in range(min(100, len(fingerprints))):
                    for j in range(i + 1, min(100, len(fingerprints))):
                        sim = DataStructs.TanimotoSimilarity(
                            fingerprints[i], fingerprints[j]
                        )
                        similarities.append(sim)

                metrics["diversity_score"] = 1 - np.mean(similarities)

            # Calculate property distributions
            properties = ["molecular_weight", "logp", "num_rings"]
            for prop in properties:
                values = [d[prop] for d in drugs if prop in d]
                if values:
                    metrics["property_distribution"][prop] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                    }

        return metrics

    def _calculate_protein_metrics(
        self, proteins: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate protein-specific quality metrics"""
        metrics = {
            "count": len(proteins),
            "unique_count": len(
                set(p.get("clean_sequence", p["sequence"]) for p in proteins)
            ),
            "diversity_score": None,
            "length_distribution": {},
        }

        if len(proteins) > 1:
            # Calculate sequence diversity (simplified)
            sequences = [p.get("clean_sequence", p["sequence"]) for p in proteins]

            # Sample for efficiency
            sample_size = min(100, len(sequences))
            sample_seqs = np.random.choice(sequences, sample_size, replace=False)

            # Calculate pairwise sequence similarity (simplified)
            similarities = []
            for i in range(len(sample_seqs)):
                for j in range(i + 1, len(sample_seqs)):
                    # Simple similarity based on common k-mers
                    seq1, seq2 = sample_seqs[i], sample_seqs[j]
                    k = 3
                    kmers1 = set(seq1[i : i + k] for i in range(len(seq1) - k + 1))
                    kmers2 = set(seq2[i : i + k] for i in range(len(seq2) - k + 1))
                    if kmers1 or kmers2:
                        similarity = len(kmers1 & kmers2) / len(kmers1 | kmers2)
                        similarities.append(similarity)

            if similarities:
                metrics["diversity_score"] = 1 - np.mean(similarities)

            # Length distribution
            lengths = [p.get("sequence_length", len(p["sequence"])) for p in proteins]
            metrics["length_distribution"] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths),
            }

        return metrics

    def _calculate_interaction_metrics(
        self, interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate interaction-specific quality metrics"""
        metrics = {
            "count": len(interactions),
            "drugs_covered": len(set(i["drug_id"] for i in interactions)),
            "proteins_covered": len(set(i["protein_id"] for i in interactions)),
            "balance_score": 0.0,
            "coverage_score": 0.0,
            "affinity_distribution": {},
        }

        if interactions and "affinity" in interactions[0]:
            affinities = [i["affinity"] for i in interactions if "affinity" in i]

            # Affinity distribution
            metrics["affinity_distribution"] = {
                "mean": np.mean(affinities),
                "std": np.std(affinities),
                "min": np.min(affinities),
                "max": np.max(affinities),
                "quartiles": np.percentile(affinities, [25, 50, 75]).tolist(),
            }

            # Balance score (how evenly distributed are the affinities)
            hist, _ = np.histogram(affinities, bins=10)
            expected_count = len(affinities) / 10
            metrics["balance_score"] = (
                1 - np.std(hist) / expected_count if expected_count > 0 else 0
            )

            # Coverage score (how well the affinity range is covered)
            theoretical_range = 6 * np.std(affinities)  # 6-sigma range
            actual_range = np.max(affinities) - np.min(affinities)
            metrics["coverage_score"] = (
                min(1.0, actual_range / theoretical_range)
                if theoretical_range > 0
                else 0
            )

        return metrics

    def _improve_data_quality(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply strategies to improve data quality"""
        logger.info("Applying data quality improvement strategies")

        # Remove outliers if configured
        if self.config["remove_outliers"]:
            interactions = self._remove_outlier_interactions(interactions)

        # Ensure minimum samples per drug/protein
        interactions = self._ensure_minimum_coverage(interactions, drugs, proteins)

        # Balance dataset if highly imbalanced
        interactions = self._balance_dataset(interactions)

        return drugs, proteins, interactions

    def _remove_outlier_interactions(
        self, interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove outlier interactions based on affinity values"""
        if not interactions or "affinity" not in interactions[0]:
            return interactions

        affinities = np.array([i["affinity"] for i in interactions])

        # Use IQR method
        Q1 = np.percentile(affinities, 25)
        Q3 = np.percentile(affinities, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.config["outlier_threshold"] * IQR
        upper_bound = Q3 + self.config["outlier_threshold"] * IQR

        clean_interactions = [
            i
            for i, a in zip(interactions, affinities)
            if lower_bound <= a <= upper_bound
        ]

        removed = len(interactions) - len(clean_interactions)
        if removed > 0:
            logger.info(f"Removed {removed} outlier interactions")

        return clean_interactions

    def _ensure_minimum_coverage(
        self,
        interactions: List[Dict[str, Any]],
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ensure minimum number of interactions per drug/protein"""
        min_samples = self.config["min_samples_per_class"]

        # Count interactions per drug and protein
        drug_counts = Counter(i["drug_id"] for i in interactions)
        protein_counts = Counter(i["protein_id"] for i in interactions)

        # Filter drugs and proteins with insufficient data
        valid_drugs = {d for d, count in drug_counts.items() if count >= min_samples}
        valid_proteins = {
            p for p, count in protein_counts.items() if count >= min_samples
        }

        # Filter interactions
        filtered = [
            i
            for i in interactions
            if i["drug_id"] in valid_drugs and i["protein_id"] in valid_proteins
        ]

        removed = len(interactions) - len(filtered)
        if removed > 0:
            logger.info(f"Removed {removed} interactions with insufficient coverage")

        return filtered

    def _balance_dataset(
        self, interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Balance dataset to avoid extreme class imbalance"""
        if not interactions or "affinity" not in interactions[0]:
            return interactions

        # Bin affinities into classes
        affinities = np.array([i["affinity"] for i in interactions])
        n_bins = 5
        bins = np.percentile(affinities, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(affinities, bins[1:-1])

        # Count samples per bin
        bin_counts = Counter(bin_indices)

        # Check if balancing is needed
        max_count = max(bin_counts.values())
        min_count = min(bin_counts.values())

        if max_count / min_count > 5:  # Significant imbalance
            logger.info(
                f"Balancing dataset (max/min ratio: {max_count / min_count:.2f})"
            )

            # Oversample minority classes
            target_count = int(np.median(list(bin_counts.values())))
            balanced_interactions = []

            for bin_idx in range(n_bins):
                bin_interactions = [
                    i for i, idx in zip(interactions, bin_indices) if idx == bin_idx
                ]

                if len(bin_interactions) < target_count:
                    # Oversample
                    sampled = np.random.choice(
                        bin_interactions, target_count, replace=True
                    ).tolist()
                    balanced_interactions.extend(sampled)
                else:
                    # Undersample
                    sampled = np.random.choice(
                        bin_interactions, target_count, replace=False
                    ).tolist()
                    balanced_interactions.extend(sampled)

            return balanced_interactions

        return interactions

    def _intelligent_sampling(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply intelligent sampling strategy instead of arbitrary reduction"""
        strategy = self.config["sampling_strategy"]

        logger.info(f"Applying {strategy} sampling strategy")

        if strategy == "stratified":
            return self._stratified_sampling(drugs, proteins, interactions)
        elif strategy == "diversity":
            return self._diversity_sampling(drugs, proteins, interactions)
        elif strategy == "cluster":
            return self._cluster_sampling(drugs, proteins, interactions)
        elif strategy == "hybrid":
            return self._hybrid_sampling(drugs, proteins, interactions)
        else:
            # No sampling
            return {"drugs": drugs, "proteins": proteins, "interactions": interactions}

    def _stratified_sampling(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Stratified sampling maintaining affinity distribution"""
        if not interactions or "affinity" not in interactions[0]:
            return {"drugs": drugs, "proteins": proteins, "interactions": interactions}

        # Determine target size (example: 80% of original)
        target_size = int(len(interactions) * 0.8)

        # Bin affinities for stratification
        affinities = np.array([i["affinity"] for i in interactions])
        n_bins = 10
        bins = np.percentile(affinities, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(affinities, bins[1:-1])

        # Stratified sampling
        sampled_indices = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_count = np.sum(bin_mask)

            if bin_count > 0:
                # Sample proportionally
                sample_count = max(1, int(target_size * bin_count / len(interactions)))
                bin_indices_list = np.where(bin_mask)[0]

                if len(bin_indices_list) <= sample_count:
                    sampled_indices.extend(bin_indices_list.tolist())
                else:
                    sampled = np.random.choice(
                        bin_indices_list, sample_count, replace=False
                    )
                    sampled_indices.extend(sampled.tolist())

        # Get sampled interactions
        sampled_interactions = [interactions[i] for i in sampled_indices]

        # Get corresponding drugs and proteins
        used_drugs = set(i["drug_id"] for i in sampled_interactions)
        used_proteins = set(i["protein_id"] for i in sampled_interactions)

        sampled_drugs = [d for d in drugs if d["id"] in used_drugs]
        sampled_proteins = [p for p in proteins if p["id"] in used_proteins]

        logger.info(
            f"Stratified sampling: {len(sampled_interactions)}/{len(interactions)} interactions"
        )

        return {
            "drugs": sampled_drugs,
            "proteins": sampled_proteins,
            "interactions": sampled_interactions,
        }

    def _diversity_sampling(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Diversity-based sampling to maximize chemical and sequence diversity"""
        # Implementation would use chemical fingerprints and sequence similarity
        # For now, returning stratified sampling as fallback
        return self._stratified_sampling(drugs, proteins, interactions)

    def _cluster_sampling(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Cluster-based sampling for representative subset"""
        # Implementation would use clustering on combined drug-protein features
        # For now, returning stratified sampling as fallback
        return self._stratified_sampling(drugs, proteins, interactions)

    def _hybrid_sampling(
        self,
        drugs: List[Dict[str, Any]],
        proteins: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Hybrid sampling combining stratification and diversity"""
        # First apply stratified sampling
        stratified = self._stratified_sampling(drugs, proteins, interactions)

        # Then apply diversity filtering if available
        if RDKIT_AVAILABLE and self.config["chemical_diversity_weight"] > 0:
            # Additional diversity-based filtering could be applied here
            pass

        return stratified

    def _create_quality_controlled_splits(
        self, sampled_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Create train/val/test splits with quality control"""
        interactions = sampled_data["interactions"]

        if not interactions:
            return {
                "train": {"drugs": [], "proteins": [], "interactions": []},
                "val": {"drugs": [], "proteins": [], "interactions": []},
                "test": {"drugs": [], "proteins": [], "interactions": []},
            }

        # Calculate split sizes
        n = len(interactions)
        test_size = int(n * self.config["test_split"])
        val_size = int(n * self.config["validation_split"])
        train_size = n - test_size - val_size

        logger.info(
            f"Creating splits: train={train_size}, val={val_size}, test={test_size}"
        )

        # Stratified split based on affinity if available
        if "affinity" in interactions[0]:
            affinities = np.array([i["affinity"] for i in interactions])

            # Bin affinities for stratification
            bins = np.percentile(affinities, [0, 33, 67, 100])
            bin_indices = np.digitize(affinities, bins[1:-1])

            # First split: train+val vs test
            indices = np.arange(n)
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=bin_indices,
                random_state=self.config["random_seed"],
            )

            # Second split: train vs val
            train_val_bins = bin_indices[train_val_idx]
            relative_val_size = val_size / (train_size + val_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                stratify=train_val_bins,
                random_state=self.config["random_seed"] + 1,
            )
        else:
            # Random split
            indices = np.random.permutation(n)
            train_idx = indices[:train_size]
            val_idx = indices[train_size : train_size + val_size]
            test_idx = indices[train_size + val_size :]

        # Create split dictionaries
        splits = {}
        for split_name, split_idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            split_interactions = [interactions[i] for i in split_idx]

            # Get corresponding drugs and proteins
            used_drugs = set(i["drug_id"] for i in split_interactions)
            used_proteins = set(i["protein_id"] for i in split_interactions)

            splits[split_name] = {
                "drugs": [d for d in sampled_data["drugs"] if d["id"] in used_drugs],
                "proteins": [
                    p for p in sampled_data["proteins"] if p["id"] in used_proteins
                ],
                "interactions": split_interactions,
            }

            # Log split statistics
            logger.info(
                f"{split_name}: {len(split_interactions)} interactions, "
                f"{len(splits[split_name]['drugs'])} drugs, "
                f"{len(splits[split_name]['proteins'])} proteins"
            )

        # Validate splits
        self._validate_splits(splits)

        return splits

    def _validate_splits(self, splits: Dict[str, Dict[str, Any]]):
        """Validate that splits maintain data quality"""
        # Check for data leakage
        train_drugs = set(d["id"] for d in splits["train"]["drugs"])
        train_proteins = set(p["id"] for p in splits["train"]["proteins"])

        # Check test set novelty
        test_drugs = set(d["id"] for d in splits["test"]["drugs"])
        test_proteins = set(p["id"] for p in splits["test"]["proteins"])

        drug_overlap = (
            len(train_drugs & test_drugs) / len(test_drugs) if test_drugs else 0
        )
        protein_overlap = (
            len(train_proteins & test_proteins) / len(test_proteins)
            if test_proteins
            else 0
        )

        logger.info(
            f"Train-test overlap: drugs={drug_overlap:.1%}, proteins={protein_overlap:.1%}"
        )

        # Check affinity distribution in
