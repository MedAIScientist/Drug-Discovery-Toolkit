#!/usr/bin/env python3
"""
Integrate ZINC endogenous metabolites and FDA drugs with existing datasets
This script merges ZINC data with ChEMBL and DAVIS datasets for comprehensive drug discovery
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

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some features will be disabled.")
    RDKIT_AVAILABLE = False


class ZINCDataIntegrator:
    """Integrate ZINC metabolites and FDA drugs with existing drug discovery datasets"""

    def __init__(
        self, zinc_properties_path: str = None, output_dir: str = "data/integrated"
    ):
        """
        Initialize the integrator

        Args:
            zinc_properties_path: Path to ZINC properties parquet file
            output_dir: Directory for integrated output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find latest ZINC properties file if not specified
        if zinc_properties_path is None:
            zinc_dir = Path("data/zinc_analysis")
            if zinc_dir.exists():
                parquet_files = list(zinc_dir.glob("zinc_properties_*.parquet"))
                if parquet_files:
                    zinc_properties_path = str(
                        max(parquet_files, key=lambda p: p.stat().st_mtime)
                    )
                    logger.info(f"Using ZINC properties: {zinc_properties_path}")

        self.zinc_properties_path = zinc_properties_path

        # Data containers
        self.zinc_data = None
        self.chembl_data = None
        self.davis_data = None
        self.integrated_data = None

        # Statistics
        self.stats = {
            "zinc_molecules": 0,
            "chembl_molecules": 0,
            "davis_molecules": 0,
            "total_unique_molecules": 0,
            "overlapping_molecules": 0,
            "unique_to_zinc": 0,
            "metabolites_integrated": 0,
            "fda_drugs_integrated": 0,
            "novel_scaffolds": 0,
            "property_enrichment": {},
        }

    def load_zinc_data(self) -> bool:
        """Load ZINC metabolites and FDA drugs data"""
        if (
            not self.zinc_properties_path
            or not Path(self.zinc_properties_path).exists()
        ):
            logger.error("ZINC properties file not found")
            return False

        try:
            self.zinc_data = pd.read_parquet(self.zinc_properties_path)
            self.stats["zinc_molecules"] = len(self.zinc_data)
            logger.info(f"Loaded {len(self.zinc_data)} ZINC molecules")

            # Count metabolites and drugs
            if "is_metabolite_like" in self.zinc_data.columns:
                self.stats["metabolites_integrated"] = int(
                    self.zinc_data["is_metabolite_like"].sum()
                )
            if "is_drug_like" in self.zinc_data.columns:
                self.stats["fda_drugs_integrated"] = int(
                    self.zinc_data["is_drug_like"].sum()
                )

            return True
        except Exception as e:
            logger.error(f"Error loading ZINC data: {e}")
            return False

    def load_chembl_data(self) -> bool:
        """Load ChEMBL dataset"""
        chembl_path = (
            "data/public_datasets/public_smiles_enriched_20251018_191725.parquet"
        )

        if not Path(chembl_path).exists():
            logger.warning("ChEMBL data not found")
            return False

        try:
            # Load only a subset of ChEMBL for memory efficiency
            self.chembl_data = pd.read_parquet(chembl_path)

            # Sample if too large
            if len(self.chembl_data) > 100000:
                logger.info("Sampling ChEMBL data for integration (100k molecules)")
                self.chembl_data = self.chembl_data.sample(n=100000, random_state=42)

            self.stats["chembl_molecules"] = len(self.chembl_data)
            logger.info(f"Loaded {len(self.chembl_data)} ChEMBL molecules")
            return True
        except Exception as e:
            logger.error(f"Error loading ChEMBL data: {e}")
            return False

    def load_davis_data(self) -> bool:
        """Load DAVIS dataset"""
        davis_path = "data/DAVIS/SMILES.txt"

        if not Path(davis_path).exists():
            logger.warning("DAVIS data not found")
            return False

        try:
            with open(davis_path, "r") as f:
                davis_json = json.load(f)

            # Convert to DataFrame
            davis_molecules = []
            for drug_id, smiles in davis_json.items():
                davis_molecules.append(
                    {"davis_id": drug_id, "smiles": smiles, "source": "DAVIS"}
                )

            self.davis_data = pd.DataFrame(davis_molecules)
            self.stats["davis_molecules"] = len(self.davis_data)
            logger.info(f"Loaded {len(self.davis_data)} DAVIS molecules")
            return True
        except Exception as e:
            logger.error(f"Error loading DAVIS data: {e}")
            return False

    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """Standardize SMILES to canonical form"""
        if not RDKIT_AVAILABLE:
            return smiles

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None

    def calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate Tanimoto similarity between two molecules"""
        if not RDKIT_AVAILABLE:
            return 0.0

        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 and mol2:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            pass

        return 0.0

    def find_overlaps(self):
        """Find overlapping molecules between datasets"""
        logger.info("Finding overlapping molecules...")

        all_smiles_sets = {}

        # Get canonical SMILES sets
        if self.zinc_data is not None:
            zinc_smiles = set(self.zinc_data["smiles"].dropna())
            all_smiles_sets["ZINC"] = zinc_smiles

        if self.chembl_data is not None:
            chembl_col = (
                "canonical_smiles"
                if "canonical_smiles" in self.chembl_data.columns
                else "smiles"
            )
            chembl_smiles = set(self.chembl_data[chembl_col].dropna())
            all_smiles_sets["ChEMBL"] = chembl_smiles

        if self.davis_data is not None:
            davis_smiles = set()
            for smiles in self.davis_data["smiles"]:
                canonical = self.standardize_smiles(smiles)
                if canonical:
                    davis_smiles.add(canonical)
            all_smiles_sets["DAVIS"] = davis_smiles

        # Calculate overlaps
        overlaps = {}

        if "ZINC" in all_smiles_sets and "ChEMBL" in all_smiles_sets:
            overlaps["ZINC_ChEMBL"] = len(
                all_smiles_sets["ZINC"].intersection(all_smiles_sets["ChEMBL"])
            )

        if "ZINC" in all_smiles_sets and "DAVIS" in all_smiles_sets:
            overlaps["ZINC_DAVIS"] = len(
                all_smiles_sets["ZINC"].intersection(all_smiles_sets["DAVIS"])
            )

        if "ChEMBL" in all_smiles_sets and "DAVIS" in all_smiles_sets:
            overlaps["ChEMBL_DAVIS"] = len(
                all_smiles_sets["ChEMBL"].intersection(all_smiles_sets["DAVIS"])
            )

        # Calculate unique molecules
        if all_smiles_sets:
            all_unique = set()
            for smiles_set in all_smiles_sets.values():
                all_unique.update(smiles_set)

            self.stats["total_unique_molecules"] = len(all_unique)

            if "ZINC" in all_smiles_sets:
                self.stats["unique_to_zinc"] = len(
                    all_smiles_sets["ZINC"]
                    - all_smiles_sets.get("ChEMBL", set())
                    - all_smiles_sets.get("DAVIS", set())
                )

        self.stats["overlapping_molecules"] = overlaps

        return overlaps

    def integrate_datasets(self):
        """Integrate all datasets into a unified collection"""
        logger.info("Integrating datasets...")

        integrated = []

        # Add ZINC data with metabolite/drug labels
        if self.zinc_data is not None:
            for _, row in self.zinc_data.iterrows():
                record = {
                    "smiles": row["smiles"],
                    "zinc_id": row.get("zinc_id", ""),
                    "source": "ZINC",
                    "is_metabolite": row.get("is_metabolite_like", False),
                    "is_fda_drug": row.get("is_drug_like", False),
                    "metabolite_class": row.get("metabolite_class", ""),
                    "drug_class": row.get("drug_class", ""),
                    "mol_weight": row.get("mol_weight", None),
                    "logp": row.get("logp", None),
                    "tpsa": row.get("tpsa", None),
                    "qed": row.get("qed", None),
                    "lipinski_violations": row.get("lipinski_violations", None),
                    "scaffold": row.get("murcko_scaffold", ""),
                }
                integrated.append(record)

        # Add unique ChEMBL molecules
        if self.chembl_data is not None:
            zinc_smiles = (
                set(self.zinc_data["smiles"]) if self.zinc_data is not None else set()
            )

            for _, row in self.chembl_data.iterrows():
                smiles = row.get("canonical_smiles", row.get("smiles", ""))

                # Skip if already in ZINC
                if smiles in zinc_smiles:
                    continue

                record = {
                    "smiles": smiles,
                    "chembl_id": row.get("chembl_id", ""),
                    "source": "ChEMBL",
                    "is_metabolite": False,
                    "is_fda_drug": False,
                    "metabolite_class": "",
                    "drug_class": "",
                    "mol_weight": row.get("mol_weight", None),
                    "logp": row.get("logp", None),
                    "tpsa": row.get("tpsa", None),
                    "qed": None,
                    "lipinski_violations": row.get("lipinski_violations", None),
                    "scaffold": row.get("scaffold", ""),
                }
                integrated.append(record)

        # Add unique DAVIS molecules
        if self.davis_data is not None:
            existing_smiles = set()
            if self.zinc_data is not None:
                existing_smiles.update(self.zinc_data["smiles"])
            if self.chembl_data is not None:
                chembl_col = (
                    "canonical_smiles"
                    if "canonical_smiles" in self.chembl_data.columns
                    else "smiles"
                )
                existing_smiles.update(self.chembl_data[chembl_col])

            for _, row in self.davis_data.iterrows():
                canonical = self.standardize_smiles(row["smiles"])

                # Skip if already exists
                if canonical in existing_smiles:
                    continue

                record = {
                    "smiles": canonical or row["smiles"],
                    "davis_id": row["davis_id"],
                    "source": "DAVIS",
                    "is_metabolite": False,
                    "is_fda_drug": False,
                    "metabolite_class": "",
                    "drug_class": "",
                    "mol_weight": None,
                    "logp": None,
                    "tpsa": None,
                    "qed": None,
                    "lipinski_violations": None,
                    "scaffold": "",
                }
                integrated.append(record)

        # Create integrated DataFrame
        self.integrated_data = pd.DataFrame(integrated)
        logger.info(f"Integrated {len(self.integrated_data)} unique molecules")

        # Calculate additional statistics
        if not self.integrated_data.empty:
            self.stats["source_distribution"] = {
                k: int(v)
                for k, v in self.integrated_data["source"]
                .value_counts()
                .to_dict()
                .items()
            }
            self.stats["metabolites_in_integrated"] = int(
                self.integrated_data["is_metabolite"].sum()
            )
            self.stats["fda_drugs_in_integrated"] = int(
                self.integrated_data["is_fda_drug"].sum()
            )

            # Count unique scaffolds
            scaffolds = self.integrated_data["scaffold"].dropna().unique()
            self.stats["total_unique_scaffolds"] = int(len(scaffolds))

    def generate_enrichment_report(self) -> Dict:
        """Generate comprehensive enrichment report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_sources": {
                "zinc_metabolites_fda": self.stats["zinc_molecules"],
                "chembl": self.stats["chembl_molecules"],
                "davis": self.stats["davis_molecules"],
            },
            "integration_statistics": {
                "total_unique_molecules": self.stats["total_unique_molecules"],
                "overlapping_molecules": self.stats["overlapping_molecules"],
                "unique_to_zinc": self.stats["unique_to_zinc"],
            },
            "biological_relevance": {
                "endogenous_metabolites": self.stats["metabolites_integrated"],
                "fda_approved_drugs": self.stats["fda_drugs_integrated"],
                "metabolites_in_integrated": self.stats.get(
                    "metabolites_in_integrated", 0
                ),
                "fda_drugs_in_integrated": self.stats.get("fda_drugs_in_integrated", 0),
            },
            "structural_diversity": {
                "unique_scaffolds": self.stats.get("total_unique_scaffolds", 0)
            },
            "source_distribution": self.stats.get("source_distribution", {}),
        }

        # Add property statistics if available
        if self.integrated_data is not None and not self.integrated_data.empty:
            property_stats = {}
            for prop in ["mol_weight", "logp", "tpsa", "qed"]:
                if prop in self.integrated_data.columns:
                    data = self.integrated_data[prop].dropna()
                    if len(data) > 0:
                        property_stats[prop] = {
                            "mean": float(data.mean()),
                            "std": float(data.std()),
                            "min": float(data.min()),
                            "max": float(data.max()),
                        }

            if property_stats:
                report["molecular_properties"] = property_stats

        return report

    def save_integrated_data(self):
        """Save integrated dataset"""
        if self.integrated_data is None or self.integrated_data.empty:
            logger.warning("No integrated data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as parquet
        parquet_file = (
            self.output_dir / f"integrated_zinc_chembl_davis_{timestamp}.parquet"
        )
        self.integrated_data.to_parquet(parquet_file, index=False)
        logger.info(f"Integrated data saved to {parquet_file}")

        # Save metabolites subset
        metabolites = self.integrated_data[
            self.integrated_data["is_metabolite"] == True
        ]
        if not metabolites.empty:
            metabolites_file = (
                self.output_dir / f"integrated_metabolites_{timestamp}.parquet"
            )
            metabolites.to_parquet(metabolites_file, index=False)
            logger.info(f"Metabolites subset saved to {metabolites_file}")

        # Save FDA drugs subset
        fda_drugs = self.integrated_data[self.integrated_data["is_fda_drug"] == True]
        if not fda_drugs.empty:
            fda_file = self.output_dir / f"integrated_fda_drugs_{timestamp}.parquet"
            fda_drugs.to_parquet(fda_file, index=False)
            logger.info(f"FDA drugs subset saved to {fda_file}")

        # Save enrichment report
        report = self.generate_enrichment_report()
        report_file = self.output_dir / f"integration_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Integration report saved to {report_file}")

        return parquet_file, report

    def run_integration(self):
        """Run the complete integration pipeline"""
        logger.info("=" * 60)
        logger.info("ZINC METABOLITES & FDA DRUGS INTEGRATION")
        logger.info("=" * 60)

        # Load datasets
        self.load_zinc_data()
        self.load_chembl_data()
        self.load_davis_data()

        # Find overlaps
        overlaps = self.find_overlaps()

        # Integrate datasets
        self.integrate_datasets()

        # Save results
        output_file, report = self.save_integrated_data()

        # Print summary
        print("\n" + "=" * 60)
        print("INTEGRATION SUMMARY")
        print("=" * 60)
        print(f"Data Sources:")
        print(f"  ZINC (metabolites & FDA): {self.stats['zinc_molecules']}")
        print(f"  ChEMBL: {self.stats['chembl_molecules']}")
        print(f"  DAVIS: {self.stats['davis_molecules']}")
        print()
        print(f"Integration Results:")
        print(f"  Total unique molecules: {self.stats['total_unique_molecules']}")
        print(f"  Unique to ZINC: {self.stats['unique_to_zinc']}")
        print(f"  Endogenous metabolites: {self.stats['metabolites_integrated']}")
        print(f"  FDA approved drugs: {self.stats['fda_drugs_integrated']}")

        if overlaps:
            print("\nOverlaps between datasets:")
            for key, count in overlaps.items():
                datasets = key.split("_")
                print(f"  {datasets[0]} ∩ {datasets[1]}: {count}")

        print()
        print(f"Enriched dataset saved to: {output_file}")
        print("=" * 60)

        return report


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate ZINC metabolites and FDA drugs with existing datasets"
    )
    parser.add_argument(
        "--zinc-data",
        help="Path to ZINC properties parquet file (auto-detect if not specified)",
    )
    parser.add_argument(
        "--output",
        default="data/integrated",
        help="Output directory for integrated data",
    )

    args = parser.parse_args()

    # Create integrator
    integrator = ZINCDataIntegrator(args.zinc_data, args.output)

    # Run integration
    report = integrator.run_integration()

    return 0


if __name__ == "__main__":
    sys.exit(main())
