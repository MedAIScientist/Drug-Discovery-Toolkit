#!/usr/bin/env python3
"""
Download and integrate public SMILES datasets from scientific databases
This script downloads SMILES data from ChEMBL, PubChem, and ZINC databases
and integrates them into the drug discovery toolkit.
"""

import os
import sys
import gzip
import json
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import pubchempy as pcp

    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some features will be disabled.")
    RDKIT_AVAILABLE = False


class PublicSMILESDownloader:
    """Download and process public SMILES datasets"""

    def __init__(self, data_dir: str = "data/public_datasets"):
        """
        Initialize the downloader

        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Drug-Discovery-Toolkit/1.0 (Academic Research)"}
        )

        # Dataset sources
        self.sources = {
            "chembl": {
                "name": "ChEMBL 36",
                "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_chemreps.txt.gz",
                "description": "ChEMBL bioactivity database - latest release",
                "size_mb": 274,
                "molecules": 2372674,
            },
            "zinc_fda": {
                "name": "ZINC FDA Approved",
                "url": "https://zinc20.docking.org/substances/subsets/fda.smi?count=all",
                "description": "FDA approved drugs from ZINC database",
                "size_mb": 1,
                "molecules": 2924,
            },
            "zinc_world": {
                "name": "ZINC World Drugs",
                "url": "https://zinc20.docking.org/substances/subsets/world.smi?count=all",
                "description": "World drug subset from ZINC database",
                "size_mb": 2,
                "molecules": 8269,
            },
            "zinc_investigational": {
                "name": "ZINC Investigational",
                "url": "https://zinc20.docking.org/substances/subsets/investigational.smi?count=all",
                "description": "Investigational drugs from ZINC",
                "size_mb": 1,
                "molecules": 3121,
            },
            "pubchem_approved": {
                "name": "PubChem FDA Approved",
                "description": "FDA approved drugs from PubChem",
                "api": True,
                "size_mb": 5,
                "molecules": 3000,
            },
        }

        self.stats = {
            "total_molecules": 0,
            "unique_smiles": 0,
            "valid_smiles": 0,
            "invalid_smiles": 0,
            "duplicates_removed": 0,
            "datasets_processed": [],
        }

    def download_file(self, url: str, filename: str, description: str = "") -> bool:
        """
        Download a file with progress bar

        Args:
            url: URL to download from
            filename: Local filename to save to
            description: Description for progress bar

        Returns:
            True if successful, False otherwise
        """
        filepath = self.data_dir / filename

        # Skip if already exists
        if filepath.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return True

        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=description or filename,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded {filename}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def download_chembl(self) -> Optional[pd.DataFrame]:
        """
        Download and process ChEMBL SMILES data

        Returns:
            DataFrame with ChEMBL molecules or None if failed
        """
        logger.info("Downloading ChEMBL dataset...")

        chembl_info = self.sources["chembl"]
        filename = "chembl_36_chemreps.txt.gz"

        if not self.download_file(chembl_info["url"], filename, "ChEMBL 36"):
            return None

        # Process the file
        logger.info("Processing ChEMBL data...")
        filepath = self.data_dir / filename

        try:
            # Read the compressed file
            df = pd.read_csv(filepath, sep="\t", compression="gzip")

            # Select relevant columns
            if "canonical_smiles" in df.columns:
                result_df = pd.DataFrame(
                    {
                        "smiles": df["canonical_smiles"],
                        "chembl_id": df.get("chembl_id", ""),
                        "name": df.get("pref_name", ""),
                        "source": "ChEMBL36",
                        "mol_weight": df.get("mw_freebase", np.nan),
                    }
                )

                # Remove invalid SMILES
                result_df = result_df[result_df["smiles"].notna()]
                result_df = result_df[result_df["smiles"] != ""]

                logger.info(f"Processed {len(result_df)} molecules from ChEMBL")
                self.stats["datasets_processed"].append("ChEMBL")
                return result_df

        except Exception as e:
            logger.error(f"Error processing ChEMBL data: {e}")

        return None

    def download_zinc_subset(self, subset: str) -> Optional[pd.DataFrame]:
        """
        Download a ZINC subset

        Args:
            subset: Name of the subset (fda, world, investigational)

        Returns:
            DataFrame with molecules or None if failed
        """
        if subset not in self.sources:
            return None

        info = self.sources[subset]
        logger.info(f"Downloading {info['name']}...")

        filename = f"{subset}.smi"

        try:
            # For ZINC, we need to handle the response differently
            response = self.session.get(info["url"])
            response.raise_for_status()

            filepath = self.data_dir / filename
            with open(filepath, "w") as f:
                f.write(response.text)

            # Process the SMILES file
            molecules = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            smiles, zinc_id = parts[0], parts[1]
                            molecules.append(
                                {
                                    "smiles": smiles,
                                    "zinc_id": zinc_id,
                                    "source": info["name"],
                                }
                            )

            df = pd.DataFrame(molecules)
            logger.info(f"Processed {len(df)} molecules from {info['name']}")
            self.stats["datasets_processed"].append(info["name"])
            return df

        except Exception as e:
            logger.error(f"Error downloading {subset}: {e}")
            return None

    def download_pubchem_subset(
        self, query: str = "FDA approved", max_records: int = 3000
    ) -> Optional[pd.DataFrame]:
        """
        Download molecules from PubChem using their API

        Args:
            query: Search query for PubChem
            max_records: Maximum number of records to retrieve

        Returns:
            DataFrame with molecules or None if failed
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available, skipping PubChem download")
            return None

        logger.info(f"Searching PubChem for: {query}")

        try:
            # Use PubChemPy to search for compounds
            # This is a simplified example - in practice, you might want to use
            # PUG-REST API directly for bulk downloads

            molecules = []

            # Get FDA approved drugs collection from PubChem
            # Using direct REST API call
            base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

            # Search for FDA approved compounds
            search_url = (
                f"{base_url}/compound/fastsubstructure/cid/JSON?list_return=listkey"
            )
            list_url = f"{base_url}/compound/listkey/{{listkey}}/cids/JSON"

            # For demonstration, we'll get a subset of bioactive compounds
            # In production, you'd want to use more specific queries

            logger.info("Note: Full PubChem download requires specific collection IDs")
            logger.info("Using sample data for demonstration")

            # Create sample data
            sample_df = pd.DataFrame(
                {"smiles": [], "pubchem_cid": [], "name": [], "source": "PubChem"}
            )

            self.stats["datasets_processed"].append("PubChem")
            return sample_df

        except Exception as e:
            logger.error(f"Error accessing PubChem: {e}")
            return None

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not RDKIT_AVAILABLE:
            # Basic validation without RDKit
            return bool(smiles and len(smiles) > 0 and not smiles.startswith("#"))

        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def calculate_properties(self, smiles: str) -> Dict:
        """
        Calculate molecular properties from SMILES

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of properties
        """
        properties = {}

        if not RDKIT_AVAILABLE:
            return properties

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                properties["mol_weight"] = Descriptors.MolWt(mol)
                properties["logp"] = Crippen.MolLogP(mol)
                properties["hbd"] = Descriptors.NumHDonors(mol)
                properties["hba"] = Descriptors.NumHAcceptors(mol)
                properties["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
                properties["tpsa"] = Descriptors.TPSA(mol)
                properties["num_rings"] = Descriptors.RingCount(mol)
                properties["num_aromatic_rings"] = Descriptors.NumAromaticRings(mol)

                # Lipinski Rule of Five
                properties["lipinski_violations"] = sum(
                    [
                        properties["mol_weight"] > 500,
                        properties["logp"] > 5,
                        properties["hbd"] > 5,
                        properties["hba"] > 10,
                    ]
                )

                # Get Murcko scaffold
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    properties["scaffold"] = (
                        Chem.MolToSmiles(scaffold) if scaffold else None
                    )
                except:
                    properties["scaffold"] = None

        except Exception as e:
            logger.debug(f"Error calculating properties for {smiles}: {e}")

        return properties

    def merge_and_deduplicate(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames and remove duplicates

        Args:
            dataframes: List of DataFrames to merge

        Returns:
            Merged and deduplicated DataFrame
        """
        logger.info("Merging and deduplicating datasets...")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        initial_count = len(combined_df)

        # Validate SMILES
        logger.info("Validating SMILES strings...")
        valid_mask = combined_df["smiles"].apply(self.validate_smiles)
        combined_df = combined_df[valid_mask]
        self.stats["invalid_smiles"] = initial_count - len(combined_df)
        self.stats["valid_smiles"] = len(combined_df)

        # Remove duplicates based on canonical SMILES
        if RDKIT_AVAILABLE:
            logger.info("Canonicalizing SMILES...")
            combined_df["canonical_smiles"] = combined_df["smiles"].apply(
                lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
                if self.validate_smiles(x)
                else None
            )
            combined_df = combined_df[combined_df["canonical_smiles"].notna()]
            combined_df = combined_df.drop_duplicates(subset=["canonical_smiles"])
        else:
            combined_df = combined_df.drop_duplicates(subset=["smiles"])

        self.stats["duplicates_removed"] = initial_count - len(combined_df)
        self.stats["unique_smiles"] = len(combined_df)

        # Calculate properties for a subset (to avoid long processing times)
        logger.info("Calculating molecular properties (sample)...")
        sample_size = min(1000, len(combined_df))
        sample_indices = np.random.choice(combined_df.index, sample_size, replace=False)

        for idx in tqdm(sample_indices, desc="Calculating properties"):
            smiles = combined_df.loc[idx, "smiles"]
            props = self.calculate_properties(smiles)
            for key, value in props.items():
                combined_df.loc[idx, key] = value

        return combined_df

    def save_dataset(self, df: pd.DataFrame, format: str = "parquet") -> str:
        """
        Save the dataset to file

        Args:
            df: DataFrame to save
            format: Output format (parquet, csv, json)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"public_smiles_enriched_{timestamp}.{format}"
        filepath = self.data_dir / filename

        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset saved to {filepath}")
        return str(filepath)

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary report of the downloaded data

        Args:
            df: Combined DataFrame

        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "dataset_summary": {
                "total_molecules": len(df),
                "unique_sources": df["source"].nunique()
                if "source" in df.columns
                else 0,
                "sources": df["source"].value_counts().to_dict()
                if "source" in df.columns
                else {},
            },
        }

        # Add property distributions if calculated
        if "mol_weight" in df.columns:
            report["property_distributions"] = {
                "mol_weight": {
                    "mean": df["mol_weight"].mean(),
                    "std": df["mol_weight"].std(),
                    "min": df["mol_weight"].min(),
                    "max": df["mol_weight"].max(),
                }
            }

        if "logp" in df.columns:
            report["property_distributions"]["logp"] = {
                "mean": df["logp"].mean(),
                "std": df["logp"].std(),
                "min": df["logp"].min(),
                "max": df["logp"].max(),
            }

        # Lipinski compliance
        if "lipinski_violations" in df.columns:
            report["drug_likeness"] = {
                "lipinski_compliant": (df["lipinski_violations"] == 0).sum(),
                "lipinski_violations_distribution": df["lipinski_violations"]
                .value_counts()
                .to_dict(),
            }

        return report

    def run(self, datasets: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete download and enrichment pipeline

        Args:
            datasets: List of dataset names to download (None for all)

        Returns:
            Tuple of (combined DataFrame, report dictionary)
        """
        logger.info("Starting public SMILES data download and enrichment")
        logger.info("=" * 60)

        if datasets is None:
            datasets = ["chembl", "zinc_fda", "zinc_world", "zinc_investigational"]

        all_dfs = []

        # Download ChEMBL
        if "chembl" in datasets:
            chembl_df = self.download_chembl()
            if chembl_df is not None:
                all_dfs.append(chembl_df)
                self.stats["total_molecules"] += len(chembl_df)

        # Download ZINC subsets
        for subset in ["zinc_fda", "zinc_world", "zinc_investigational"]:
            if subset in datasets:
                zinc_df = self.download_zinc_subset(subset)
                if zinc_df is not None:
                    all_dfs.append(zinc_df)
                    self.stats["total_molecules"] += len(zinc_df)

        # Download PubChem (if included)
        if "pubchem" in datasets:
            pubchem_df = self.download_pubchem_subset()
            if pubchem_df is not None:
                all_dfs.append(pubchem_df)
                self.stats["total_molecules"] += len(pubchem_df)

        if not all_dfs:
            logger.error("No datasets could be downloaded")
            return pd.DataFrame(), {}

        # Merge and process
        combined_df = self.merge_and_deduplicate(all_dfs)

        # Generate report
        report = self.generate_report(combined_df)

        # Save results
        output_file = self.save_dataset(combined_df, "parquet")
        report["output_file"] = output_file

        # Save report
        report_file = (
            self.data_dir
            / f"enrichment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("Download and enrichment completed successfully!")
        logger.info(f"Total unique molecules: {len(combined_df)}")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Report saved to: {report_file}")

        return combined_df, report


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and enrich public SMILES datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["chembl", "zinc_fda", "zinc_world", "zinc_investigational", "pubchem"],
        help="Specific datasets to download (default: all except pubchem)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/public_datasets",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json"],
        default="parquet",
        help="Output format for the combined dataset",
    )

    args = parser.parse_args()

    # Create downloader instance
    downloader = PublicSMILESDownloader(args.output_dir)

    # Run the download pipeline
    combined_df, report = downloader.run(args.datasets)

    # Print summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total molecules processed: {report['statistics']['total_molecules']}")
    print(f"Unique molecules: {report['statistics']['unique_smiles']}")
    print(f"Invalid SMILES removed: {report['statistics']['invalid_smiles']}")
    print(f"Duplicates removed: {report['statistics']['duplicates_removed']}")
    print(
        f"Datasets processed: {', '.join(report['statistics']['datasets_processed'])}"
    )

    if "property_distributions" in report:
        print("\nMolecular Property Ranges:")
        for prop, stats in report["property_distributions"].items():
            print(
                f"  {prop}: {stats['min']:.2f} - {stats['max']:.2f} (mean: {stats['mean']:.2f})"
            )

    if "drug_likeness" in report:
        print(
            f"\nLipinski Rule of Five Compliant: {report['drug_likeness']['lipinski_compliant']}"
        )

    print("\nOutput files:")
    print(f"  Data: {report['output_file']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
