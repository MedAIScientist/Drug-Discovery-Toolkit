#!/usr/bin/env python3
"""
Analyze and integrate ZINC endogenous human metabolites and FDA approved drugs
This script performs detailed analysis of metabolites and drugs for drug discovery enrichment
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
import logging
import hashlib
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, Fragments
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem, DataStructs, Draw
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.ML.Cluster import Butina
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform

    RDKIT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RDKit or dependencies not fully available: {e}")
    RDKIT_AVAILABLE = False


class ZINCEndogenousFDAAnalyzer:
    """Analyze ZINC endogenous metabolites and FDA approved drugs"""

    def __init__(self, input_file: str, output_dir: str = "data/zinc_analysis"):
        """
        Initialize the analyzer

        Args:
            input_file: Path to ZINC JSON file
            output_dir: Directory for output files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.molecules = []
        self.valid_molecules = []
        self.metabolites = []
        self.fda_drugs = []
        self.properties_df = None

        # Analysis results
        self.stats = {
            "total_molecules": 0,
            "valid_smiles": 0,
            "invalid_smiles": 0,
            "unique_scaffolds": 0,
            "metabolite_like": 0,
            "drug_like": 0,
            "both_metabolite_drug": 0,
            "molecular_weight_range": {},
            "logp_range": {},
            "functional_groups": {},
            "clustering_results": {},
        }

        # Known metabolite patterns
        self.metabolite_patterns = {
            "amino_acids": ["N[C@@H]", "N[C@H]", "C(=O)O", "C(N)C(=O)O"],
            "sugars": ["O[C@H]", "O[C@@H]", "[C@H](O)", "[C@@H](O)", "CO[C@H]"],
            "nucleotides": ["n1cnc2", "n1ccc(N)nc1", "O[C@H]1[C@@H]"],
            "fatty_acids": ["CCCC", "C(=O)O", "CC/C=C", "CCCCC"],
            "steroids": ["C1CC2C3CCC4", "[C@]12", "[C@@]12", "CC[C@H]"],
            "vitamins": ["c1cccnc1", "Cc1cc2nc", "C=C1CC"],
            "neurotransmitters": ["NCCc1ccc(O)", "CNC[C@H](O)c1ccc", "c1ccc2[nH]"],
            "organic_acids": ["C(=O)O", "CC(=O)O", "O=C(O)C"],
        }

        # FDA drug patterns
        self.drug_patterns = {
            "beta_lactam": ["C1C(=O)N[C@H]1", "C1C(=O)N[C@@H]1"],
            "sulfonamide": ["S(=O)(=O)N", "NS(=O)(=O)"],
            "fluoroquinolone": ["c1cc2c(cc1F)", "C(=O)c1cn"],
            "benzodiazepine": ["c1ccc2NC(=O)CNc2c1", "c1ccc2N=C"],
            "steroid": ["C1CC2C3CCC4", "[C@]12", "[C@@]12"],
            "phenothiazine": ["c1ccc2Sc3ccccc3Nc2c1"],
        }

    def load_data(self) -> bool:
        """
        Load and parse ZINC JSON data

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading data from {self.input_file}")

            with open(self.input_file, "r") as f:
                self.molecules = json.load(f)

            self.stats["total_molecules"] = len(self.molecules)
            logger.info(f"Loaded {len(self.molecules)} molecules")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def validate_and_standardize(self):
        """Validate SMILES and standardize molecules"""
        logger.info("Validating and standardizing molecules...")

        for mol_data in tqdm(self.molecules, desc="Validating"):
            zinc_id = mol_data.get("zinc_id", "")
            smiles = mol_data.get("smiles", "")

            if not RDKIT_AVAILABLE:
                self.valid_molecules.append(mol_data)
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Standardize
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    inchi = Chem.MolToInchi(mol)
                    inchi_key = Chem.MolToInchiKey(mol)

                    valid_mol = {
                        "zinc_id": zinc_id,
                        "original_smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        "inchi": inchi,
                        "inchi_key": inchi_key,
                        "mol": mol,
                    }

                    self.valid_molecules.append(valid_mol)
                    self.stats["valid_smiles"] += 1
                else:
                    self.stats["invalid_smiles"] += 1
                    logger.debug(f"Invalid SMILES: {smiles}")

            except Exception as e:
                self.stats["invalid_smiles"] += 1
                logger.debug(f"Error processing {zinc_id}: {e}")

    def calculate_properties(self):
        """Calculate molecular properties for all valid molecules"""
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping property calculation")
            return

        logger.info("Calculating molecular properties...")

        properties = []

        for mol_data in tqdm(self.valid_molecules, desc="Calculating properties"):
            mol = mol_data["mol"]
            zinc_id = mol_data["zinc_id"]

            try:
                props = {
                    "zinc_id": zinc_id,
                    "smiles": mol_data["canonical_smiles"],
                    # Basic descriptors
                    "mol_weight": Descriptors.MolWt(mol),
                    "logp": Crippen.MolLogP(mol),
                    "logd": Crippen.MolMR(mol),
                    "tpsa": Descriptors.TPSA(mol),
                    # Counts
                    "num_atoms": mol.GetNumAtoms(),
                    "num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
                    "num_bonds": mol.GetNumBonds(),
                    "num_heteroatoms": Descriptors.NumHeteroatoms(mol),
                    # H-bond donors/acceptors
                    "hbd": Descriptors.NumHDonors(mol),
                    "hba": Descriptors.NumHAcceptors(mol),
                    # Rotatable bonds
                    "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                    # Rings
                    "num_rings": Descriptors.RingCount(mol),
                    "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                    "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
                    "num_saturated_rings": Descriptors.NumSaturatedRings(mol),
                    # Complexity
                    "bertz_ct": Descriptors.BertzCT(mol),
                    "mol_complexity": len(Chem.MolToSmiles(mol)),
                    # Lipinski Rule of Five
                    "lipinski_violations": sum(
                        [
                            Descriptors.MolWt(mol) > 500,
                            Crippen.MolLogP(mol) > 5,
                            Descriptors.NumHDonors(mol) > 5,
                            Descriptors.NumHAcceptors(mol) > 10,
                        ]
                    ),
                    # Veber's Rule (oral bioavailability)
                    "veber_violations": sum(
                        [
                            Descriptors.NumRotatableBonds(mol) > 10,
                            Descriptors.TPSA(mol) > 140,
                        ]
                    ),
                    # QED (Quantitative Estimate of Drug-likeness)
                    "qed": self.calculate_qed(mol),
                    # Functional groups
                    "num_amines": Fragments.fr_NH2(mol)
                    + Fragments.fr_NH1(mol)
                    + Fragments.fr_NH0(mol),
                    "num_carboxylic_acids": Fragments.fr_COO(mol)
                    + Fragments.fr_COO2(mol),
                    "num_alcohols": Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
                    "num_esters": Fragments.fr_ester(mol),
                    "num_ethers": Fragments.fr_ether(mol),
                    "num_ketones": Fragments.fr_ketone(mol),
                    "num_aldehydes": Fragments.fr_aldehyde(mol),
                    "num_sulfur": Fragments.fr_sulfide(mol) + Fragments.fr_SH(mol),
                    "num_halogen": Fragments.fr_halogen(mol),
                    # Scaffold
                    "murcko_scaffold": self.get_murcko_scaffold(mol),
                    # Classification
                    "is_metabolite_like": self.is_metabolite_like(mol_data),
                    "is_drug_like": self.is_drug_like(mol),
                    "metabolite_class": self.classify_metabolite(mol_data),
                    "drug_class": self.classify_drug(mol_data),
                }

                properties.append(props)

                # Update statistics
                if props["is_metabolite_like"]:
                    self.stats["metabolite_like"] += 1
                    self.metabolites.append(mol_data)

                if props["is_drug_like"]:
                    self.stats["drug_like"] += 1
                    self.fda_drugs.append(mol_data)

                if props["is_metabolite_like"] and props["is_drug_like"]:
                    self.stats["both_metabolite_drug"] += 1

            except Exception as e:
                logger.debug(f"Error calculating properties for {zinc_id}: {e}")

        # Create DataFrame
        self.properties_df = pd.DataFrame(properties)

        # Calculate statistics
        if not self.properties_df.empty:
            self.stats["molecular_weight_range"] = {
                "min": float(self.properties_df["mol_weight"].min()),
                "max": float(self.properties_df["mol_weight"].max()),
                "mean": float(self.properties_df["mol_weight"].mean()),
                "std": float(self.properties_df["mol_weight"].std()),
            }

            self.stats["logp_range"] = {
                "min": float(self.properties_df["logp"].min()),
                "max": float(self.properties_df["logp"].max()),
                "mean": float(self.properties_df["logp"].mean()),
                "std": float(self.properties_df["logp"].std()),
            }

            # Count unique scaffolds
            scaffolds = self.properties_df["murcko_scaffold"].dropna().unique()
            self.stats["unique_scaffolds"] = len(scaffolds)

    def calculate_qed(self, mol) -> float:
        """
        Calculate QED (Quantitative Estimate of Drug-likeness)
        Simplified version based on Bickerton et al. 2012
        """
        try:
            from rdkit.Chem import QED

            return QED.qed(mol)
        except:
            # Simplified QED calculation if QED module not available
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotb = Descriptors.NumRotatableBonds(mol)

            # Normalize to 0-1 scale (simplified)
            qed = 1.0
            qed *= np.exp(-(((mw - 300) / 100) ** 2)) if mw > 300 else 1
            qed *= np.exp(-(((abs(logp - 2.5)) / 2) ** 2))
            qed *= np.exp(-((hbd / 5) ** 2)) if hbd > 5 else 1
            qed *= np.exp(-((hba / 10) ** 2)) if hba > 10 else 1
            qed *= np.exp(-(((tpsa - 60) / 40) ** 2)) if tpsa > 60 else 1
            qed *= np.exp(-((rotb / 8) ** 2)) if rotb > 8 else 1

            return min(1.0, max(0.0, qed))

    def get_murcko_scaffold(self, mol) -> Optional[str]:
        """Extract Murcko scaffold from molecule"""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold:
                return Chem.MolToSmiles(scaffold)
        except:
            pass
        return None

    def is_metabolite_like(self, mol_data) -> bool:
        """Determine if molecule is metabolite-like"""
        if not RDKIT_AVAILABLE:
            return False

        mol = mol_data["mol"]
        smiles = mol_data["canonical_smiles"]

        # Check molecular weight (metabolites are typically smaller)
        mw = Descriptors.MolWt(mol)
        if mw > 700:  # Most metabolites are < 700 Da
            return False

        # Check for metabolite patterns
        metabolite_score = 0
        for pattern_type, patterns in self.metabolite_patterns.items():
            for pattern in patterns:
                if pattern in smiles:
                    metabolite_score += 1
                    break

        # Check for common metabolite functional groups
        if Fragments.fr_COO(mol) > 0:  # Carboxylic acid
            metabolite_score += 2
        if Fragments.fr_NH2(mol) > 0:  # Primary amine
            metabolite_score += 1
        if Fragments.fr_Al_OH(mol) > 0:  # Alcohol
            metabolite_score += 1

        # Check for sugar-like properties
        oxygen_ratio = Descriptors.NumHeteroatoms(mol) / mol.GetNumAtoms()
        if oxygen_ratio > 0.3 and Fragments.fr_Al_OH(mol) >= 3:
            metabolite_score += 3

        return metabolite_score >= 3

    def is_drug_like(self, mol) -> bool:
        """Determine if molecule is drug-like using multiple rules"""
        if not RDKIT_AVAILABLE:
            return True  # Default to true if we can't calculate

        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        lipinski_pass = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

        # Veber's Rule
        rotb = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        veber_pass = rotb <= 10 and tpsa <= 140

        # Additional drug-like criteria
        num_rings = Descriptors.RingCount(mol)
        drug_like_rings = 1 <= num_rings <= 6

        # QED score
        qed = self.calculate_qed(mol)
        good_qed = qed > 0.5

        # Consider drug-like if passes most criteria
        criteria_passed = sum([lipinski_pass, veber_pass, drug_like_rings, good_qed])

        return criteria_passed >= 2

    def classify_metabolite(self, mol_data) -> str:
        """Classify metabolite into categories"""
        if not RDKIT_AVAILABLE:
            return "unknown"

        mol = mol_data["mol"]
        smiles = mol_data["canonical_smiles"]

        # Check for amino acids
        if "N[C@@H](" in smiles or "N[C@H](" in smiles:
            if "C(=O)O" in smiles:
                return "amino_acid"

        # Check for sugars
        if mol.GetNumAtoms() < 30:
            oh_count = Fragments.fr_Al_OH(mol)
            if oh_count >= 3 and "O[C@" in smiles:
                return "sugar"

        # Check for nucleotides/nucleosides
        if "n1cnc2" in smiles or "n1ccc(N)nc1" in smiles:
            return "nucleotide"

        # Check for fatty acids
        if "CCCC" in smiles and "C(=O)O" in smiles:
            carbon_chain = max(len(s) for s in smiles.split("C(=O)O")[0].split("C"))
            if carbon_chain >= 4:
                return "fatty_acid"

        # Check for steroids
        if "C1CC2C3CCC4" in smiles or "[C@]12CC" in smiles:
            return "steroid"

        # Check for neurotransmitters
        if "NCCc1ccc(O)" in smiles or "CNC[C@H](O)c1ccc" in smiles:
            return "neurotransmitter"

        # Check for organic acids
        acid_count = Fragments.fr_COO(mol) + Fragments.fr_COO2(mol)
        if acid_count > 0 and mol.GetNumAtoms() < 20:
            return "organic_acid"

        # Check for vitamins
        if "c1cccnc1" in smiles or "Cc1cc2nc" in smiles:
            return "vitamin"

        return "other_metabolite"

    def classify_drug(self, mol_data) -> str:
        """Classify drug into therapeutic categories"""
        if not RDKIT_AVAILABLE:
            return "unknown"

        mol = mol_data["mol"]
        smiles = mol_data["canonical_smiles"]

        # Check for common drug scaffolds
        for drug_class, patterns in self.drug_patterns.items():
            for pattern in patterns:
                if pattern in smiles:
                    return drug_class

        # Additional classification based on properties
        mw = Descriptors.MolWt(mol)

        if "CN1C" in smiles or "CN(C)" in smiles:
            return "cns_drug"  # CNS drugs often have tertiary amines

        if Fragments.fr_halogen(mol) > 0:
            return "halogenated_drug"

        if mw > 400 and Descriptors.NumAromaticRings(mol) >= 2:
            return "large_molecule_drug"

        return "small_molecule_drug"

    def perform_clustering(self, n_clusters: int = 10):
        """Perform molecular clustering"""
        if not RDKIT_AVAILABLE or not self.valid_molecules:
            logger.warning("Cannot perform clustering")
            return

        logger.info("Performing molecular clustering...")

        # Generate fingerprints
        fps = []
        for mol_data in self.valid_molecules[
            : min(1000, len(self.valid_molecules))
        ]:  # Limit for performance
            mol = mol_data["mol"]
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)

        # Calculate distance matrix
        n = len(fps)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distance = 1 - similarity
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method="average")

        # Store clustering results
        self.stats["clustering_results"] = {
            "num_molecules_clustered": n,
            "method": "hierarchical_average_linkage",
            "distance_metric": "tanimoto",
        }

        return linkage_matrix, distance_matrix

    def generate_visualizations(self):
        """Generate visualization plots"""
        if self.properties_df is None or self.properties_df.empty:
            logger.warning("No data for visualization")
            return

        logger.info("Generating visualizations...")

        # Set up the plot style
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(
            "ZINC Endogenous Metabolites & FDA Drugs Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Molecular Weight Distribution
        ax = axes[0, 0]
        ax.hist(
            self.properties_df["mol_weight"],
            bins=50,
            color="blue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Molecular Weight (Da)")
        ax.set_ylabel("Count")
        ax.set_title("Molecular Weight Distribution")
        ax.axvline(x=500, color="r", linestyle="--", label="Lipinski limit")
        ax.legend()

        # 2. LogP Distribution
        ax = axes[0, 1]
        ax.hist(
            self.properties_df["logp"],
            bins=50,
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("LogP")
        ax.set_ylabel("Count")
        ax.set_title("LogP Distribution")
        ax.axvline(x=5, color="r", linestyle="--", label="Lipinski limit")
        ax.legend()

        # 3. TPSA Distribution
        ax = axes[0, 2]
        ax.hist(
            self.properties_df["tpsa"],
            bins=50,
            color="orange",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("TPSA (Ų)")
        ax.set_ylabel("Count")
        ax.set_title("Topological Polar Surface Area")
        ax.axvline(x=140, color="r", linestyle="--", label="Veber limit")
        ax.legend()

        # 4. H-bond Donors vs Acceptors
        ax = axes[1, 0]
        ax.scatter(
            self.properties_df["hbd"],
            self.properties_df["hba"],
            alpha=0.5,
            c=self.properties_df["qed"],
            cmap="viridis",
        )
        ax.set_xlabel("H-bond Donors")
        ax.set_ylabel("H-bond Acceptors")
        ax.set_title("H-bond Donors vs Acceptors (colored by QED)")
        ax.axvline(x=5, color="r", linestyle="--", alpha=0.5)
        ax.axhline(y=10, color="r", linestyle="--", alpha=0.5)

        # 5. Metabolite vs Drug Classification
        ax = axes[1, 1]
        categories = ["Metabolite-like", "Drug-like", "Both", "Neither"]
        counts = [
            self.stats["metabolite_like"] - self.stats["both_metabolite_drug"],
            self.stats["drug_like"] - self.stats["both_metabolite_drug"],
            self.stats["both_metabolite_drug"],
            len(self.properties_df)
            - self.stats["metabolite_like"]
            - self.stats["drug_like"]
            + self.stats["both_metabolite_drug"],
        ]
        colors = ["lightblue", "lightgreen", "gold", "lightcoral"]
        ax.pie(
            counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90
        )
        ax.set_title("Molecule Classification")

        # 6. QED Distribution
        ax = axes[1, 2]
        ax.hist(
            self.properties_df["qed"],
            bins=50,
            color="purple",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("QED Score")
        ax.set_ylabel("Count")
        ax.set_title("Quantitative Estimate of Drug-likeness")
        ax.axvline(x=0.5, color="r", linestyle="--", label="QED threshold")
        ax.legend()

        # 7. Metabolite Classes
        ax = axes[2, 0]
        metabolite_counts = (
            self.properties_df["metabolite_class"].value_counts().head(10)
        )
        ax.barh(metabolite_counts.index, metabolite_counts.values, color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Metabolite Classes")
        ax.invert_yaxis()

        # 8. Drug Classes
        ax = axes[2, 1]
        drug_counts = self.properties_df["drug_class"].value_counts().head(10)
        ax.barh(drug_counts.index, drug_counts.values, color="darkgreen")
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Drug Classes")
        ax.invert_yaxis()

        # 9. Lipinski & Veber Violations
        ax = axes[2, 2]
        violations = pd.DataFrame(
            {
                "Lipinski": self.properties_df["lipinski_violations"]
                .value_counts()
                .sort_index(),
                "Veber": self.properties_df["veber_violations"]
                .value_counts()
                .sort_index(),
            }
        )
        violations.plot(kind="bar", ax=ax, color=["coral", "skyblue"])
        ax.set_xlabel("Number of Violations")
        ax.set_ylabel("Count")
        ax.set_title("Rule Violations Distribution")
        ax.legend()

        plt.tight_layout()

        # Save the figure
        output_file = (
            self.output_dir
            / f"zinc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Visualizations saved to {output_file}")

        plt.close()

    def compare_with_existing(self, existing_data_path: str = None):
        """Compare with existing ChEMBL or DAVIS datasets"""
        comparison_results = {
            "overlap_count": 0,
            "unique_to_zinc": 0,
            "unique_scaffolds_zinc": 0,
            "property_comparison": {},
        }

        if existing_data_path and Path(existing_data_path).exists():
            try:
                # Load existing data (assumed to be parquet)
                existing_df = pd.read_parquet(existing_data_path)

                if "canonical_smiles" in existing_df.columns:
                    existing_smiles = set(existing_df["canonical_smiles"].dropna())
                    zinc_smiles = set(self.properties_df["smiles"].dropna())

                    overlap = existing_smiles.intersection(zinc_smiles)
                    comparison_results["overlap_count"] = len(overlap)
                    comparison_results["unique_to_zinc"] = len(
                        zinc_smiles - existing_smiles
                    )

                    logger.info(
                        f"Found {len(overlap)} overlapping molecules with existing dataset"
                    )

            except Exception as e:
                logger.error(f"Error comparing with existing data: {e}")

        return comparison_results

    def export_enriched_data(self):
        """Export enriched data in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save properties DataFrame
        if self.properties_df is not None:
            # Parquet format
            parquet_file = self.output_dir / f"zinc_properties_{timestamp}.parquet"
            self.properties_df.to_parquet(parquet_file, index=False)
            logger.info(f"Properties saved to {parquet_file}")

            # CSV format
            csv_file = self.output_dir / f"zinc_properties_{timestamp}.csv"
            self.properties_df.to_csv(csv_file, index=False)
            logger.info(f"Properties saved to {csv_file}")

            # Separate metabolites and drugs
            metabolites_df = self.properties_df[
                self.properties_df["is_metabolite_like"]
            ]
            drugs_df = self.properties_df[self.properties_df["is_drug_like"]]

            if not metabolites_df.empty:
                metabolites_file = (
                    self.output_dir / f"zinc_metabolites_{timestamp}.parquet"
                )
                metabolites_df.to_parquet(metabolites_file, index=False)
                logger.info(f"Metabolites saved to {metabolites_file}")

            if not drugs_df.empty:
                drugs_file = self.output_dir / f"zinc_drugs_{timestamp}.parquet"
                drugs_df.to_parquet(drugs_file, index=False)
                logger.info(f"FDA drugs saved to {drugs_file}")

        # Save analysis report
        report = self.generate_report()
        report_file = self.output_dir / f"zinc_analysis_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Analysis report saved to {report_file}")

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_file": str(self.input_file),
            "statistics": self.stats,
            "dataset_overview": {
                "total_molecules": self.stats["total_molecules"],
                "valid_molecules": self.stats["valid_smiles"],
                "invalid_molecules": self.stats["invalid_smiles"],
                "validation_rate": f"{(self.stats['valid_smiles'] / self.stats['total_molecules'] * 100):.2f}%"
                if self.stats["total_molecules"] > 0
                else "0%",
            },
            "classification": {
                "metabolite_like": self.stats["metabolite_like"],
                "drug_like": self.stats["drug_like"],
                "both_metabolite_and_drug": self.stats["both_metabolite_drug"],
                "metabolite_only": self.stats["metabolite_like"]
                - self.stats["both_metabolite_drug"],
                "drug_only": self.stats["drug_like"]
                - self.stats["both_metabolite_drug"],
            },
            "molecular_properties": {
                "mol_weight": self.stats.get("molecular_weight_range", {}),
                "logp": self.stats.get("logp_range", {}),
            },
            "structural_diversity": {
                "unique_scaffolds": self.stats["unique_scaffolds"],
            },
        }

        # Add detailed property distributions if available
        if self.properties_df is not None and not self.properties_df.empty:
            report["property_distributions"] = {}

            for prop in [
                "mol_weight",
                "logp",
                "tpsa",
                "hbd",
                "hba",
                "rotatable_bonds",
                "qed",
            ]:
                if prop in self.properties_df.columns:
                    data = self.properties_df[prop].dropna()
                    if len(data) > 0:
                        report["property_distributions"][prop] = {
                            "mean": float(data.mean()),
                            "std": float(data.std()),
                            "min": float(data.min()),
                            "max": float(data.max()),
                            "q25": float(data.quantile(0.25)),
                            "median": float(data.median()),
                            "q75": float(data.quantile(0.75)),
                        }

            # Add metabolite class distribution
            if "metabolite_class" in self.properties_df.columns:
                metabolite_dist = (
                    self.properties_df["metabolite_class"].value_counts().to_dict()
                )
                report["metabolite_classes"] = {
                    k: int(v) for k, v in metabolite_dist.items()
                }

            # Add drug class distribution
            if "drug_class" in self.properties_df.columns:
                drug_dist = self.properties_df["drug_class"].value_counts().to_dict()
                report["drug_classes"] = {k: int(v) for k, v in drug_dist.items()}

            # Add rule compliance
            report["rule_compliance"] = {
                "lipinski_compliant": int(
                    (self.properties_df["lipinski_violations"] == 0).sum()
                ),
                "veber_compliant": int(
                    (self.properties_df["veber_violations"] == 0).sum()
                ),
                "qed_above_0.5": int((self.properties_df["qed"] > 0.5).sum()),
            }

        return report

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("=" * 60)
        logger.info("ZINC ENDOGENOUS METABOLITES & FDA DRUGS ANALYSIS")
        logger.info("=" * 60)

        # Load data
        if not self.load_data():
            logger.error("Failed to load data")
            return None

        # Validate and standardize
        self.validate_and_standardize()

        # Calculate properties
        self.calculate_properties()

        # Perform clustering
        try:
            self.perform_clustering()
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")

        # Generate visualizations
        try:
            self.generate_visualizations()
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        # Compare with existing datasets
        chembl_path = (
            "data/public_datasets/public_smiles_enriched_20251018_191725.parquet"
        )
        if Path(chembl_path).exists():
            comparison = self.compare_with_existing(chembl_path)
            self.stats["chembl_comparison"] = comparison

        # Export enriched data
        self.export_enriched_data()

        # Generate final report
        report = self.generate_report()

        return report


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze ZINC endogenous metabolites and FDA approved drugs"
    )
    parser.add_argument(
        "--input",
        default="data/public_datasets/endogenous+fda.json",
        help="Input JSON file from ZINC",
    )
    parser.add_argument(
        "--output",
        default="data/zinc_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--compare-with",
        help="Path to existing dataset for comparison (e.g., ChEMBL)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = ZINCEndogenousFDAAnalyzer(args.input, args.output)

    # Run analysis
    report = analyzer.run_analysis()

    if report:
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total molecules: {report['dataset_overview']['total_molecules']}")
        print(f"Valid molecules: {report['dataset_overview']['valid_molecules']}")
        print(f"Validation rate: {report['dataset_overview']['validation_rate']}")
        print()
        print("Classification:")
        print(f"  Metabolite-like: {report['classification']['metabolite_like']}")
        print(f"  Drug-like: {report['classification']['drug_like']}")
        print(f"  Both: {report['classification']['both_metabolite_and_drug']}")
        print(
            f"  Unique scaffolds: {report['structural_diversity']['unique_scaffolds']}"
        )
        print()

        if "property_distributions" in report:
            print("Molecular Properties (mean ± std):")
            for prop, stats in report["property_distributions"].items():
                if prop in ["mol_weight", "logp", "tpsa", "qed"]:
                    print(f"  {prop}: {stats['mean']:.2f} ± {stats['std']:.2f}")

        if "metabolite_classes" in report:
            print("\nTop Metabolite Classes:")
            sorted_metabolites = sorted(
                report["metabolite_classes"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for cls, count in sorted_metabolites:
                print(f"  {cls}: {count}")

        if "drug_classes" in report:
            print("\nTop Drug Classes:")
            sorted_drugs = sorted(
                report["drug_classes"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for cls, count in sorted_drugs:
                print(f"  {cls}: {count}")

        if "rule_compliance" in report:
            print("\nRule Compliance:")
            total = report["dataset_overview"]["valid_molecules"]
            for rule, count in report["rule_compliance"].items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {rule}: {count} ({percentage:.1f}%)")

        print("=" * 60)

        # Save summary to text file
        summary_file = (
            Path(args.output)
            / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(summary_file, "w") as f:
            f.write(json.dumps(report, indent=2, default=str))

        print(f"\nFull report saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
