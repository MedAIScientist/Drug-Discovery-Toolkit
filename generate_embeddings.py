#!/usr/bin/env python3
"""
Generate ChemBERTa embeddings for all drugs with SMILES in the database
"""

import os
import sys
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from arango import ArangoClient
    from dotenv import load_dotenv
    from transformers import AutoTokenizer, AutoModel
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    load_dotenv()
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    LIBRARIES_AVAILABLE = False
    sys.exit(1)


class DrugEmbeddingGenerator:
    """Generate and store ChemBERTa embeddings for drug molecules"""

    def __init__(self):
        self.db = None
        self.tokenizer = None
        self.model = None
        self.drug_collection = None
        self.stats = {
            "total_drugs": 0,
            "drugs_with_smiles": 0,
            "embeddings_generated": 0,
            "embeddings_failed": 0,
            "properties_calculated": 0,
        }

        self._init_database()
        self._init_model()

    def _init_database(self):
        """Initialize ArangoDB connection"""
        try:
            ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
            ARANGO_USER = os.getenv("ARANGO_USER", "root")
            ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

            client = ArangoClient(hosts=ARANGO_HOST)
            self.db = client.db("NeuThera", username=ARANGO_USER, password=ARANGO_PASS)
            self.drug_collection = self.db.collection("drugs")

            logger.info("Successfully connected to NeuThera database")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            sys.exit(1)

    def _init_model(self):
        """Initialize ChemBERTa model"""
        try:
            logger.info("Loading ChemBERTa model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1"
            )
            self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"ChemBERTa model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            sys.exit(1)

    def generate_embedding(self, smiles: str):
        """Generate ChemBERTa embedding for a SMILES string"""
        if not smiles or not isinstance(smiles, str):
            return None

        try:
            # Tokenize and generate embedding
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling over sequence dimension
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return embedding[0].tolist()

        except Exception as e:
            logger.debug(f"Embedding generation failed for {smiles}: {e}")
            return None

    def calculate_molecular_properties(self, smiles: str):
        """Calculate molecular properties using RDKit"""
        properties = {}

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return properties

            # Calculate basic molecular descriptors
            properties["molecular_weight"] = Descriptors.ExactMolWt(mol)
            properties["logp"] = Descriptors.MolLogP(mol)
            properties["num_rings"] = Chem.rdMolDescriptors.CalcNumRings(mol)
            properties["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
            properties["num_h_donors"] = Descriptors.NumHDonors(mol)
            properties["num_h_acceptors"] = Descriptors.NumHAcceptors(mol)
            properties["tpsa"] = Descriptors.TPSA(mol)
            properties["num_heavy_atoms"] = mol.GetNumHeavyAtoms()

            # Canonical SMILES
            properties["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

        except Exception as e:
            logger.debug(f"Property calculation failed for {smiles}: {e}")

        return properties

    def process_drugs(self, batch_size=100):
        """Process all drugs in batches"""
        logger.info("Starting drug processing...")

        # Get all drugs
        cursor = self.db.aql.execute("FOR d IN drugs RETURN d", batch_size=batch_size)

        drugs_to_update = []

        for drug in tqdm(cursor, desc="Processing drugs"):
            self.stats["total_drugs"] += 1

            # Check if drug has SMILES
            smiles = drug.get("smiles") or drug.get("canonical_smiles")

            if not smiles:
                # Try to get SMILES from drug_id if it looks like SMILES
                drug_id = drug.get("drug_id", "")
                if drug_id and ("C" in drug_id or "c" in drug_id) and len(drug_id) > 5:
                    # Might be a SMILES string
                    mol = Chem.MolFromSmiles(drug_id)
                    if mol:
                        smiles = drug_id

            if smiles:
                self.stats["drugs_with_smiles"] += 1

                # Skip if already has embedding
                if drug.get("embedding") and len(drug.get("embedding", [])) > 0:
                    logger.debug(f"Drug {drug['_key']} already has embedding, skipping")
                    continue

                # Generate embedding
                embedding = self.generate_embedding(smiles)

                if embedding:
                    self.stats["embeddings_generated"] += 1

                    # Calculate molecular properties
                    properties = self.calculate_molecular_properties(smiles)
                    if properties:
                        self.stats["properties_calculated"] += 1

                    # Prepare update document
                    update_doc = {
                        "_key": drug["_key"],
                        "embedding": embedding,
                        "smiles": smiles,
                        "embedding_generated_at": datetime.now().isoformat(),
                    }

                    # Add molecular properties
                    update_doc.update(properties)

                    drugs_to_update.append(update_doc)
                else:
                    self.stats["embeddings_failed"] += 1

            # Update database in batches
            if len(drugs_to_update) >= batch_size:
                self._update_drugs_batch(drugs_to_update)
                drugs_to_update = []

        # Update remaining drugs
        if drugs_to_update:
            self._update_drugs_batch(drugs_to_update)

        logger.info("Drug processing complete!")
        return self.stats

    def _update_drugs_batch(self, drugs):
        """Update a batch of drugs in the database"""
        try:
            for drug in drugs:
                self.drug_collection.update({"_key": drug["_key"]}, drug, merge=True)
            logger.info(f"Updated {len(drugs)} drugs with embeddings")
        except Exception as e:
            logger.error(f"Failed to update drugs: {e}")

    def add_smiles_from_chembl(self):
        """Add SMILES from ChEMBL data if available"""
        chembl_file = Path("./data/chembl_35_chemreps.txt.gz")

        if not chembl_file.exists():
            logger.warning("ChEMBL data file not found")
            return

        logger.info("Loading ChEMBL SMILES data...")

        try:
            import gzip
            import pandas as pd

            with gzip.open(chembl_file, "rt") as f:
                # Read ChEMBL data
                chembl_data = pd.read_csv(f, sep="\t", low_memory=False)

                # Get columns with SMILES
                if "canonical_smiles" in chembl_data.columns:
                    smiles_col = "canonical_smiles"
                elif "smiles" in chembl_data.columns:
                    smiles_col = "smiles"
                else:
                    logger.warning("No SMILES column found in ChEMBL data")
                    return

                # Get ID column
                if "chembl_id" in chembl_data.columns:
                    id_col = "chembl_id"
                elif "molregno" in chembl_data.columns:
                    id_col = "molregno"
                else:
                    id_col = chembl_data.columns[0]

                logger.info(f"Processing {len(chembl_data)} ChEMBL compounds...")

                # Add ChEMBL drugs to database
                batch = []
                for idx, row in tqdm(
                    chembl_data.iterrows(),
                    total=len(chembl_data),
                    desc="Adding ChEMBL drugs",
                ):
                    if pd.notna(row[smiles_col]):
                        drug_doc = {
                            "_key": f"chembl_{row[id_col]}".replace(":", "_").replace(
                                "/", "_"
                            ),
                            "drug_id": str(row[id_col]),
                            "smiles": str(row[smiles_col]),
                            "source": "ChEMBL",
                        }
                        batch.append(drug_doc)

                    if len(batch) >= 1000:
                        try:
                            self.drug_collection.insert_many(batch, overwrite=True)
                        except:
                            pass
                        batch = []

                # Insert remaining
                if batch:
                    try:
                        self.drug_collection.insert_many(batch, overwrite=True)
                    except:
                        pass

                logger.info("ChEMBL drugs added to database")

        except Exception as e:
            logger.error(f"Failed to process ChEMBL data: {e}")

    def generate_report(self):
        """Generate a summary report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "database": "NeuThera",
            "model": "seyonec/ChemBERTa-zinc-base-v1",
            "device": str(self.device),
            "success_rate": (
                self.stats["embeddings_generated"]
                / self.stats["drugs_with_smiles"]
                * 100
            )
            if self.stats["drugs_with_smiles"] > 0
            else 0,
        }

        # Save report
        report_file = f"embedding_generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total drugs processed: {self.stats['total_drugs']:,}")
        print(f"Drugs with SMILES: {self.stats['drugs_with_smiles']:,}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']:,}")
        print(f"Embeddings failed: {self.stats['embeddings_failed']:,}")
        print(f"Properties calculated: {self.stats['properties_calculated']:,}")
        print(f"Success rate: {report['success_rate']:.2f}%")
        print("=" * 60 + "\n")

        return report


def main():
    """Main execution function"""
    generator = DrugEmbeddingGenerator()

    # First try to add more drugs from ChEMBL
    generator.add_smiles_from_chembl()

    # Process all drugs and generate embeddings
    stats = generator.process_drugs(batch_size=100)

    # Generate report
    report = generator.generate_report()

    return report


if __name__ == "__main__":
    if not LIBRARIES_AVAILABLE:
        print("ERROR: Required libraries not installed")
        sys.exit(1)

    main()
