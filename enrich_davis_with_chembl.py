#!/usr/bin/env python3
"""
Enrich DAVIS dataset with ChEMBL public SMILES data
This script combines DAVIS drug-target interaction data with ChEMBL molecular properties
"""

import os
import sys
import json
import pickle
import gzip
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
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some features will be disabled.")
    RDKIT_AVAILABLE = False


class DAVISChEMBLEnricher:
    """Enrich DAVIS dataset with ChEMBL molecular data"""

    def __init__(
        self,
        davis_dir: str = "data/DAVIS",
        chembl_file: str = None,
        output_dir: str = "data/enriched",
    ):
        """
        Initialize the enricher

        Args:
            davis_dir: Directory containing DAVIS dataset
            chembl_file: Path to ChEMBL parquet file
            output_dir: Directory for output files
        """
        self.davis_dir = Path(davis_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find ChEMBL file if not specified
        if chembl_file is None:
            public_dir = Path("data/public_datasets")
            if public_dir.exists():
                parquet_files = list(
                    public_dir.glob("public_smiles_enriched_*.parquet")
                )
                if parquet_files:
                    chembl_file = str(
                        max(parquet_files, key=lambda p: p.stat().st_mtime)
                    )
                    logger.info(f"Using ChEMBL file: {chembl_file}")

        self.chembl_file = chembl_file

        # Data containers
        self.davis_drugs = {}
        self.davis_targets = {}
        self.davis_interactions = None
        self.chembl_data = None
        self.enriched_drugs = {}

        # Statistics
        self.stats = {
            "davis_drugs": 0,
            "davis_targets": 0,
            "davis_interactions": 0,
            "chembl_molecules": 0,
            "matched_drugs": 0,
            "new_properties": 0,
            "similar_molecules_found": 0,
            "enrichment_rate": 0.0,
        }

    def load_davis_data(self) -> bool:
        """
        Load DAVIS dataset

        Returns:
            True if successful, False otherwise
        """
        logger.info("Loading DAVIS dataset...")

        try:
            # Load drug SMILES
            drug_file = self.davis_dir / "drug.txt"
            if drug_file.exists():
                with open(drug_file, "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        drug_id, smiles = line.strip().split()
                        self.davis_drugs[drug_id] = {
                            "smiles": smiles,
                            "davis_id": drug_id,
                            "index": i,
                        }
                self.stats["davis_drugs"] = len(self.davis_drugs)
                logger.info(f"Loaded {len(self.davis_drugs)} DAVIS drugs")

            # Load target sequences
            target_file = self.davis_dir / "target.txt"
            if target_file.exists():
                with open(target_file, "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            target_id = parts[0]
                            sequence = parts[1] if len(parts) > 1 else ""
                            self.davis_targets[target_id] = {
                                "sequence": sequence,
                                "davis_id": target_id,
                                "index": i,
                            }
                self.stats["davis_targets"] = len(self.davis_targets)
                logger.info(f"Loaded {len(self.davis_targets)} DAVIS targets")

            # Load interaction matrix
            interaction_file = self.davis_dir / "affinity.txt"
            if interaction_file.exists():
                self.davis_interactions = np.loadtxt(interaction_file)
                self.stats["davis_interactions"] = self.davis_interactions.size
                logger.info(
                    f"Loaded interaction matrix: {self.davis_interactions.shape}"
                )

            return True

        except Exception as e:
            logger.error(f"Error loading DAVIS data: {e}")
            return False

    def load_chembl_data(self) -> bool:
        """
        Load ChEMBL dataset

        Returns:
            True if successful, False otherwise
        """
        if not self.chembl_file or not Path(self.chembl_file).exists():
            logger.error("ChEMBL file not found")
            return False

        try:
            logger.info("Loading ChEMBL data...")
            self.chembl_data = pd.read_parquet(self.chembl_file)
            self.stats["chembl_molecules"] = len(self.chembl_data)
            logger.info(f"Loaded {len(self.chembl_data)} ChEMBL molecules")

            # Create SMILES index for faster lookup
            if "canonical_smiles" in self.chembl_data.columns:
                self.chembl_data.set_index("canonical_smiles", drop=False, inplace=True)
            elif "smiles" in self.chembl_data.columns:
                self.chembl_data.set_index("smiles", drop=False, inplace=True)

            return True

        except Exception as e:
            logger.error(f"Error loading ChEMBL data: {e}")
            return False

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES or None if invalid
        """
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
        """
        Calculate Tanimoto similarity between two molecules

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string

        Returns:
            Similarity score (0-1)
        """
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

    def find_similar_molecules(self, smiles: str, threshold: float = 0.7) -> List[Dict]:
        """
        Find similar molecules in ChEMBL dataset

        Args:
            smiles: Query SMILES string
            threshold: Similarity threshold

        Returns:
            List of similar molecules with properties
        """
        similar = []

        if not RDKIT_AVAILABLE or self.chembl_data is None:
            return similar

        try:
            query_mol = Chem.MolFromSmiles(smiles)
            if not query_mol:
                return similar

            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)

            # Sample ChEMBL data to avoid processing all molecules
            sample_size = min(10000, len(self.chembl_data))
            sample = self.chembl_data.sample(n=sample_size, random_state=42)

            for idx, row in sample.iterrows():
                try:
                    ref_smiles = row.get("smiles", idx)
                    ref_mol = Chem.MolFromSmiles(ref_smiles)

                    if ref_mol:
                        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
                        similarity = DataStructs.TanimotoSimilarity(query_fp, ref_fp)

                        if similarity >= threshold:
                            similar.append(
                                {
                                    "smiles": ref_smiles,
                                    "similarity": similarity,
                                    "chembl_id": row.get("chembl_id", ""),
                                    "name": row.get("name", ""),
                                    "mol_weight": row.get("mol_weight", None),
                                    "logp": row.get("logp", None),
                                }
                            )
                except:
                    continue

            # Sort by similarity
            similar.sort(key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            logger.debug(f"Error finding similar molecules: {e}")

        return similar[:10]  # Return top 10 most similar

    def enrich_drug(self, drug_id: str, drug_info: Dict) -> Dict:
        """
        Enrich a single drug with ChEMBL data

        Args:
            drug_id: DAVIS drug ID
            drug_info: Drug information dictionary

        Returns:
            Enriched drug information
        """
        enriched = drug_info.copy()
        smiles = drug_info["smiles"]

        # Canonicalize SMILES
        canonical = self.canonicalize_smiles(smiles)
        if canonical:
            enriched["canonical_smiles"] = canonical

            # Look for exact match in ChEMBL
            if self.chembl_data is not None and canonical in self.chembl_data.index:
                chembl_match = self.chembl_data.loc[canonical]

                # Add ChEMBL properties
                enriched["chembl_match"] = True
                enriched["chembl_id"] = chembl_match.get("chembl_id", "")
                enriched["chembl_name"] = chembl_match.get("name", "")

                # Add molecular properties
                for prop in [
                    "mol_weight",
                    "logp",
                    "hbd",
                    "hba",
                    "tpsa",
                    "rotatable_bonds",
                    "num_rings",
                    "num_aromatic_rings",
                    "lipinski_violations",
                    "scaffold",
                ]:
                    if prop in chembl_match and pd.notna(chembl_match[prop]):
                        enriched[prop] = chembl_match[prop]
                        self.stats["new_properties"] += 1

                self.stats["matched_drugs"] += 1
            else:
                enriched["chembl_match"] = False

                # Find similar molecules
                similar = self.find_similar_molecules(canonical)
                if similar:
                    enriched["similar_molecules"] = similar
                    self.stats["similar_molecules_found"] += 1

                    # Use properties from most similar molecule
                    if similar[0]["similarity"] > 0.85:
                        most_similar = similar[0]
                        enriched["inferred_from_similar"] = True
                        enriched["similarity_score"] = most_similar["similarity"]

                        for prop in ["mol_weight", "logp"]:
                            if prop in most_similar and most_similar[prop]:
                                enriched[f"inferred_{prop}"] = most_similar[prop]

        # Calculate additional properties if RDKit is available
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    enriched["num_atoms"] = mol.GetNumAtoms()
                    enriched["num_bonds"] = mol.GetNumBonds()
                    enriched["molecular_formula"] = (
                        Chem.rdMolDescriptors.CalcMolFormula(mol)
                    )

                    # Calculate if not already present
                    if "mol_weight" not in enriched:
                        enriched["mol_weight"] = Descriptors.MolWt(mol)
                    if "logp" not in enriched:
                        enriched["logp"] = Crippen.MolLogP(mol)
                    if "tpsa" not in enriched:
                        enriched["tpsa"] = Descriptors.TPSA(mol)
            except:
                pass

        return enriched

    def run_enrichment(self):
        """Run the complete enrichment pipeline"""
        logger.info("=" * 60)
        logger.info("DAVIS-ChEMBL ENRICHMENT PIPELINE")
        logger.info("=" * 60)

        # Load datasets
        if not self.load_davis_data():
            logger.error("Failed to load DAVIS data")
            return

        if not self.load_chembl_data():
            logger.error("Failed to load ChEMBL data")
            return

        # Enrich each drug
        logger.info("Enriching DAVIS drugs with ChEMBL data...")

        for drug_id, drug_info in tqdm(
            self.davis_drugs.items(), desc="Enriching drugs"
        ):
            self.enriched_drugs[drug_id] = self.enrich_drug(drug_id, drug_info)

        # Calculate enrichment rate
        if self.stats["davis_drugs"] > 0:
            self.stats["enrichment_rate"] = (
                self.stats["matched_drugs"] / self.stats["davis_drugs"]
            ) * 100

        # Save enriched data
        self.save_enriched_data()

        # Generate and save report
        self.generate_report()

    def save_enriched_data(self):
        """Save enriched drug data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = self.output_dir / f"davis_drugs_enriched_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.enriched_drugs, f, indent=2, default=str)
        logger.info(f"Enriched drugs saved to {json_file}")

        # Save as DataFrame
        df = pd.DataFrame.from_dict(self.enriched_drugs, orient="index")
        parquet_file = self.output_dir / f"davis_drugs_enriched_{timestamp}.parquet"
        df.to_parquet(parquet_file)
        logger.info(f"Enriched drugs saved to {parquet_file}")

        # Save interaction matrix with drug properties
        if self.davis_interactions is not None:
            np.save(
                self.output_dir / f"davis_interactions_{timestamp}.npy",
                self.davis_interactions,
            )

    def generate_report(self):
        """Generate and save enrichment report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "summary": {
                "total_davis_drugs": self.stats["davis_drugs"],
                "total_chembl_molecules": self.stats["chembl_molecules"],
                "exact_matches": self.stats["matched_drugs"],
                "similar_molecules_found": self.stats["similar_molecules_found"],
                "enrichment_rate": f"{self.stats['enrichment_rate']:.2f}%",
                "new_properties_added": self.stats["new_properties"],
            },
        }

        # Add property distribution if available
        if self.enriched_drugs:
            df = pd.DataFrame.from_dict(self.enriched_drugs, orient="index")

            property_stats = {}
            for prop in ["mol_weight", "logp", "tpsa", "lipinski_violations"]:
                if prop in df.columns:
                    valid_data = df[prop].dropna()
                    if len(valid_data) > 0:
                        property_stats[prop] = {
                            "count": len(valid_data),
                            "mean": float(valid_data.mean()),
                            "std": float(valid_data.std()),
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                        }

            if property_stats:
                report["property_distributions"] = property_stats

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"enrichment_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("ENRICHMENT SUMMARY")
        print("=" * 60)
        print(f"DAVIS drugs: {self.stats['davis_drugs']}")
        print(f"ChEMBL molecules: {self.stats['chembl_molecules']}")
        print(f"Exact matches: {self.stats['matched_drugs']}")
        print(f"Similar molecules found: {self.stats['similar_molecules_found']}")
        print(f"Enrichment rate: {self.stats['enrichment_rate']:.2f}%")
        print(f"New properties added: {self.stats['new_properties']}")
        print("=" * 60)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich DAVIS dataset with ChEMBL molecular data"
    )
    parser.add_argument(
        "--davis-dir", default="data/DAVIS", help="Directory containing DAVIS dataset"
    )
    parser.add_argument(
        "--chembl-file",
        help="Path to ChEMBL parquet file (auto-detect if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/enriched",
        help="Output directory for enriched data",
    )

    args = parser.parse_args()

    # Create enricher
    enricher = DAVISChEMBLEnricher(
        davis_dir=args.davis_dir,
        chembl_file=args.chembl_file,
        output_dir=args.output_dir,
    )

    # Run enrichment
    enricher.run_enrichment()

    return 0


if __name__ == "__main__":
    sys.exit(main())
