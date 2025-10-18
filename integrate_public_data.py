#!/usr/bin/env python3
"""
Integrate downloaded public SMILES datasets with existing drug discovery database
This script enriches the existing database with new molecular data from public sources
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from arango import ArangoClient
    from dotenv import load_dotenv
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    from transformers import AutoTokenizer, AutoModel
    import torch

    load_dotenv()
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some libraries not available: {e}")
    LIBRARIES_AVAILABLE = False


class PublicDataIntegrator:
    """Integrate public SMILES data with existing database"""

    def __init__(self, public_data_path: str = None):
        """
        Initialize the integrator

        Args:
            public_data_path: Path to the public dataset parquet file
        """
        self.public_data_path = public_data_path
        self.db = None
        self.drug_collection = None
        self.enrichment_collection = None
        self.device = None
        self.tokenizer = None
        self.model = None

        # Statistics
        self.stats = {
            "existing_drugs": 0,
            "new_drugs_added": 0,
            "drugs_enriched": 0,
            "duplicates_found": 0,
            "errors": 0,
            "properties_added": 0,
            "embeddings_generated": 0,
        }

        if LIBRARIES_AVAILABLE:
            self._connect_database()
            self._setup_ml_models()

    def _connect_database(self):
        """Connect to ArangoDB database"""
        try:
            client = ArangoClient(
                hosts=os.getenv("ARANGO_HOST", "http://localhost:8529")
            )
            self.db = client.db(
                os.getenv("ARANGO_DATABASE", "drug_discovery"),
                username=os.getenv("ARANGO_USERNAME", "root"),
                password=os.getenv("ARANGO_PASSWORD", ""),
            )

            # Get or create collections
            if not self.db.has_collection("drugs"):
                self.drug_collection = self.db.create_collection("drugs")
            else:
                self.drug_collection = self.db.collection("drugs")

            if not self.db.has_collection("enrichment_log"):
                self.enrichment_collection = self.db.create_collection("enrichment_log")
            else:
                self.enrichment_collection = self.db.collection("enrichment_log")

            logger.info("Successfully connected to database")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db = None

    def _setup_ml_models(self):
        """Setup ML models for embedding generation"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "allenai/scibert_scivocab_uncased"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            logger.info(f"ML models loaded on {self.device}")

        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
            self.tokenizer = None
            self.model = None

    def load_public_data(self) -> pd.DataFrame:
        """
        Load the downloaded public dataset

        Returns:
            DataFrame with public data
        """
        if not self.public_data_path:
            # Find the latest public dataset
            data_dir = Path("data/public_datasets")
            if data_dir.exists():
                parquet_files = list(data_dir.glob("public_smiles_enriched_*.parquet"))
                if parquet_files:
                    self.public_data_path = str(
                        max(parquet_files, key=lambda p: p.stat().st_mtime)
                    )
                    logger.info(f"Using latest public dataset: {self.public_data_path}")

        if not self.public_data_path or not Path(self.public_data_path).exists():
            logger.error("No public dataset found")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.public_data_path)
            logger.info(f"Loaded {len(df)} molecules from public dataset")
            return df
        except Exception as e:
            logger.error(f"Error loading public data: {e}")
            return pd.DataFrame()

    def get_existing_smiles(self) -> set:
        """
        Get set of existing SMILES in database

        Returns:
            Set of canonical SMILES strings
        """
        if not self.db:
            return set()

        try:
            # Query all existing SMILES
            cursor = self.db.aql.execute(
                "FOR drug IN drugs RETURN drug.canonical_smiles", batch_size=10000
            )
            existing = set(smiles for smiles in cursor if smiles)
            self.stats["existing_drugs"] = len(existing)
            logger.info(f"Found {len(existing)} existing drugs in database")
            return existing

        except Exception as e:
            logger.error(f"Error getting existing SMILES: {e}")
            return set()

    def calculate_fingerprint(self, smiles: str) -> Optional[str]:
        """
        Calculate molecular fingerprint

        Args:
            smiles: SMILES string

        Returns:
            Fingerprint as hex string or None
        """
        if not LIBRARIES_AVAILABLE:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # Convert to hex string for storage
                arr = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr.tobytes().hex()
        except:
            return None

    def generate_embedding(
        self, smiles: str, description: str = ""
    ) -> Optional[List[float]]:
        """
        Generate embedding for molecule

        Args:
            smiles: SMILES string
            description: Optional text description

        Returns:
            Embedding vector or None
        """
        if not self.tokenizer or not self.model:
            return None

        try:
            # Combine SMILES and description
            text = f"Molecule: {smiles}"
            if description:
                text += f" Description: {description}"

            # Tokenize and generate embedding
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                )

            return embedding.tolist()

        except Exception as e:
            logger.debug(f"Error generating embedding: {e}")
            return None

    def prepare_drug_document(self, row: pd.Series) -> Dict:
        """
        Prepare a drug document for database insertion

        Args:
            row: DataFrame row with drug data

        Returns:
            Document dictionary
        """
        # Generate unique ID
        smiles = row.get("smiles", "")
        doc_id = hashlib.md5(smiles.encode()).hexdigest()

        # Basic document structure
        doc = {
            "_key": doc_id,
            "smiles": smiles,
            "canonical_smiles": row.get("canonical_smiles", smiles),
            "source": row.get("source", "public_dataset"),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Add identifiers
        if "chembl_id" in row and pd.notna(row["chembl_id"]):
            doc["chembl_id"] = row["chembl_id"]
        if "zinc_id" in row and pd.notna(row["zinc_id"]):
            doc["zinc_id"] = row["zinc_id"]
        if "name" in row and pd.notna(row["name"]):
            doc["name"] = row["name"]

        # Add calculated properties
        properties = {}
        prop_fields = [
            "mol_weight",
            "logp",
            "hbd",
            "hba",
            "rotatable_bonds",
            "tpsa",
            "num_rings",
            "num_aromatic_rings",
            "lipinski_violations",
        ]

        for field in prop_fields:
            if field in row and pd.notna(row[field]):
                properties[field] = float(row[field])

        if properties:
            doc["properties"] = properties
            self.stats["properties_added"] += len(properties)

        # Add scaffold if available
        if "scaffold" in row and pd.notna(row["scaffold"]):
            doc["scaffold"] = row["scaffold"]

        # Calculate fingerprint
        fingerprint = self.calculate_fingerprint(smiles)
        if fingerprint:
            doc["fingerprint"] = fingerprint

        # Generate embedding
        embedding = self.generate_embedding(smiles, row.get("name", ""))
        if embedding:
            doc["embedding"] = embedding
            self.stats["embeddings_generated"] += 1

        return doc

    def integrate_data(self, batch_size: int = 1000):
        """
        Integrate public data into database

        Args:
            batch_size: Number of documents to process in each batch
        """
        logger.info("Starting data integration...")

        # Load public data
        public_df = self.load_public_data()
        if public_df.empty:
            logger.error("No data to integrate")
            return

        # Get existing SMILES
        existing_smiles = self.get_existing_smiles()

        # Filter out existing molecules (if checking for duplicates)
        if "canonical_smiles" in public_df.columns:
            new_mask = ~public_df["canonical_smiles"].isin(existing_smiles)
            new_df = public_df[new_mask]
            self.stats["duplicates_found"] = len(public_df) - len(new_df)
        else:
            new_df = public_df

        logger.info(f"Processing {len(new_df)} new molecules...")

        if self.db and not new_df.empty:
            # Process in batches
            total_batches = (len(new_df) + batch_size - 1) // batch_size

            for i in tqdm(
                range(0, len(new_df), batch_size),
                total=total_batches,
                desc="Integrating",
            ):
                batch_df = new_df.iloc[i : i + batch_size]
                documents = []

                for _, row in batch_df.iterrows():
                    try:
                        doc = self.prepare_drug_document(row)
                        documents.append(doc)
                    except Exception as e:
                        logger.debug(f"Error preparing document: {e}")
                        self.stats["errors"] += 1

                # Insert batch into database
                if documents:
                    try:
                        result = self.drug_collection.insert_many(
                            documents, overwrite=False, silent=True
                        )
                        self.stats["new_drugs_added"] += len(documents)
                    except Exception as e:
                        logger.error(f"Error inserting batch: {e}")
                        self.stats["errors"] += len(documents)

        # Log enrichment
        self._log_enrichment()

    def _log_enrichment(self):
        """Log the enrichment process"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_file": self.public_data_path,
            "statistics": self.stats,
            "status": "completed",
        }

        # Save to database if available
        if self.db and self.enrichment_collection:
            try:
                self.enrichment_collection.insert(log_entry)
                logger.info("Enrichment logged to database")
            except Exception as e:
                logger.error(f"Error logging enrichment: {e}")

        # Save to file
        log_file = f"data/public_datasets/integration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        logger.info(f"Integration log saved to {log_file}")

    def generate_report(self) -> Dict:
        """
        Generate integration report

        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "database_status": "connected" if self.db else "not connected",
            "ml_models_status": "loaded" if self.model else "not loaded",
        }

        # Calculate success rate
        total_attempted = self.stats["new_drugs_added"] + self.stats["errors"]
        if total_attempted > 0:
            report["success_rate"] = (
                self.stats["new_drugs_added"] / total_attempted
            ) * 100

        return report

    def run(self):
        """Run the complete integration pipeline"""
        logger.info("=" * 60)
        logger.info("PUBLIC DATA INTEGRATION PIPELINE")
        logger.info("=" * 60)

        # Integrate data
        self.integrate_data()

        # Generate report
        report = self.generate_report()

        # Print summary
        print("\n" + "=" * 60)
        print("INTEGRATION SUMMARY")
        print("=" * 60)
        print(f"Existing drugs in database: {self.stats['existing_drugs']}")
        print(f"New drugs added: {self.stats['new_drugs_added']}")
        print(f"Duplicates found: {self.stats['duplicates_found']}")
        print(f"Properties added: {self.stats['properties_added']}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']}")
        print(f"Errors: {self.stats['errors']}")

        if "success_rate" in report:
            print(f"Success rate: {report['success_rate']:.2f}%")

        print("=" * 60)

        return report


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate public SMILES data with existing database"
    )
    parser.add_argument(
        "--data-file",
        help="Path to public dataset parquet file (auto-detect if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database operations",
    )

    args = parser.parse_args()

    # Create integrator
    integrator = PublicDataIntegrator(args.data_file)

    # Run integration
    report = integrator.run()

    # Save final report
    report_file = f"data/public_datasets/integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Final report saved to {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
