#!/usr/bin/env python3
"""
Enrich drug database with SMILES from DAVIS and DrugBank datasets
This script loads SMILES data from multiple sources and updates the drug collection
"""

import os
import sys
import json
import gzip
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
from typing import Dict, List, Optional
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
    from rdkit.Chem import Descriptors, AllChem
    from transformers import AutoTokenizer, AutoModel
    import torch

    load_dotenv()
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    LIBRARIES_AVAILABLE = False


class DrugDataEnricher:
    """Enrich drug database with SMILES and molecular properties"""

    def __init__(self):
        self.db = None
        self.drug_collection = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.stats = {
            "total_drugs": 0,
            "drugs_enriched": 0,
            "smiles_added": 0,
            "embeddings_generated": 0,
            "davis_drugs_added": 0,
            "drugbank_drugs_matched": 0,
            "properties_calculated": 0,
        }

        self._init_database()
        self._init_models()

    def _init_database(self):
        """Initialize database connection"""
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

    def _init_models(self):
        """Initialize ChemBERTa model for embeddings"""
        try:
            logger.info("Loading ChemBERTa model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1"
            )
            self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"ChemBERTa model loaded on {self.device}")
        except Exception as e:
            logger.warning(f"ChemBERTa model initialization failed: {e}")
            self.model = None

    def generate_embedding(self, smiles: str) -> Optional[List[float]]:
        """Generate ChemBERTa embedding for SMILES"""
        if not self.model or not smiles:
            return None

        try:
            inputs = self.tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embedding[0].tolist()
        except Exception as e:
            logger.debug(f"Embedding generation failed for {smiles}: {e}")
            return None

    def calculate_properties(self, smiles: str) -> Dict:
        """Calculate molecular properties from SMILES"""
        properties = {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                properties["molecular_weight"] = Descriptors.ExactMolWt(mol)
                properties["logp"] = Descriptors.MolLogP(mol)
                properties["num_rings"] = Chem.rdMolDescriptors.CalcNumRings(mol)
                properties["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
                properties["num_h_donors"] = Descriptors.NumHDonors(mol)
                properties["num_h_acceptors"] = Descriptors.NumHAcceptors(mol)
                properties["tpsa"] = Descriptors.TPSA(mol)
                properties["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
                properties["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

                # Lipinski's Rule of Five
                properties["lipinski_violations"] = sum(
                    [
                        properties["molecular_weight"] > 500,
                        properties["logp"] > 5,
                        properties["num_h_donors"] > 5,
                        properties["num_h_acceptors"] > 10,
                    ]
                )

                self.stats["properties_calculated"] += 1
        except Exception as e:
            logger.debug(f"Property calculation failed: {e}")

        return properties

    def load_davis_data(self):
        """Load and process DAVIS dataset"""
        logger.info("Loading DAVIS dataset...")

        davis_dir = Path("./data/DAVIS")
        if not davis_dir.exists():
            logger.warning("DAVIS directory not found")
            return

        try:
            # Load SMILES
            smiles_file = davis_dir / "SMILES.txt"
            if smiles_file.exists():
                with open(smiles_file, "r") as f:
                    smiles_list = [line.strip() for line in f if line.strip()]

                logger.info(f"Loaded {len(smiles_list)} SMILES from DAVIS")

                # Load target sequences
                target_file = davis_dir / "target_seq.txt"
                targets = []
                if target_file.exists():
                    with open(target_file, "r") as f:
                        targets = [line.strip() for line in f if line.strip()]

                # Load affinity matrix
                affinity_file = davis_dir / "affinity.txt"
                affinity_matrix = None
                if affinity_file.exists():
                    affinity_matrix = np.loadtxt(affinity_file)
                    logger.info(
                        f"Loaded affinity matrix with shape {affinity_matrix.shape}"
                    )

                # Process and add DAVIS drugs
                for i, smiles in enumerate(
                    tqdm(smiles_list, desc="Processing DAVIS drugs")
                ):
                    drug_key = f"davis_drug_{i}"

                    # Check if drug already exists
                    existing = self.drug_collection.get(drug_key)

                    drug_doc = {
                        "_key": drug_key,
                        "drug_id": drug_key,
                        "smiles": smiles,
                        "source": "DAVIS",
                        "davis_index": i,
                    }

                    # Calculate properties
                    properties = self.calculate_properties(smiles)
                    drug_doc.update(properties)

                    # Generate embedding
                    embedding = self.generate_embedding(smiles)
                    if embedding:
                        drug_doc["embedding"] = embedding
                        self.stats["embeddings_generated"] += 1

                    # Store in database
                    if existing:
                        self.drug_collection.update(drug_doc, merge=True)
                    else:
                        self.drug_collection.insert(drug_doc, overwrite=True)
                        self.stats["davis_drugs_added"] += 1

                # Add drug-protein interactions if affinity matrix exists
                if affinity_matrix is not None and targets:
                    self._add_davis_interactions(smiles_list, targets, affinity_matrix)

                logger.info(f"Added {self.stats['davis_drugs_added']} DAVIS drugs")

        except Exception as e:
            logger.error(f"Failed to load DAVIS data: {e}")

    def _add_davis_interactions(
        self, drugs: List[str], targets: List[str], affinity_matrix: np.ndarray
    ):
        """Add DAVIS drug-protein interactions"""
        try:
            # Ensure protein collection exists
            if not self.db.has_collection("proteins"):
                self.db.create_collection("proteins")

            protein_collection = self.db.collection("proteins")

            # Add proteins
            for j, seq in enumerate(targets):
                protein_doc = {
                    "_key": f"davis_protein_{j}",
                    "protein_id": f"davis_protein_{j}",
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "source": "DAVIS",
                }
                protein_collection.insert(protein_doc, overwrite=True)

            # Ensure drug_protein edge collection exists
            if not self.db.has_collection("drug_protein"):
                self.db.create_collection("drug_protein", edge=True)

            edge_collection = self.db.collection("drug_protein")

            # Add high-affinity interactions (top 10%)
            threshold = np.percentile(affinity_matrix, 90)

            interactions_added = 0
            for i in range(min(len(drugs), affinity_matrix.shape[0])):
                for j in range(min(len(targets), affinity_matrix.shape[1])):
                    affinity = affinity_matrix[i, j]

                    if affinity >= threshold:
                        edge_doc = {
                            "_from": f"drugs/davis_drug_{i}",
                            "_to": f"proteins/davis_protein_{j}",
                            "affinity": float(affinity),
                            "type": "drug_target_interaction",
                            "source": "DAVIS",
                        }
                        try:
                            edge_collection.insert(edge_doc, overwrite=False)
                            interactions_added += 1
                        except:
                            pass

            logger.info(
                f"Added {interactions_added} high-affinity drug-protein interactions from DAVIS"
            )

        except Exception as e:
            logger.error(f"Failed to add DAVIS interactions: {e}")

    def load_drugbank_vocabulary(self):
        """Load DrugBank vocabulary and match with existing drugs"""
        logger.info("Loading DrugBank vocabulary...")

        drugbank_file = Path("./data/drugbank_all_drugbank_vocabulary.csv.zip")
        if not drugbank_file.exists():
            logger.warning("DrugBank vocabulary file not found")
            return

        try:
            # Extract and read DrugBank data
            with zipfile.ZipFile(drugbank_file, "r") as zip_ref:
                with zip_ref.open("drugbank vocabulary.csv") as f:
                    df = pd.read_csv(f)

            logger.info(f"Loaded {len(df)} DrugBank entries")

            # Create mapping of DrugBank IDs to SMILES
            drugbank_smiles = {}

            for col in df.columns:
                if "smiles" in col.lower():
                    for idx, row in df.iterrows():
                        db_id = (
                            row.get("DrugBank ID")
                            or row.get("drugbank_id")
                            or row.get("ID")
                        )
                        smiles = row.get(col)

                        if db_id and smiles and pd.notna(smiles):
                            drugbank_smiles[str(db_id)] = str(smiles)
                    break

            logger.info(f"Found {len(drugbank_smiles)} DrugBank entries with SMILES")

            # Update existing drugs with SMILES
            cursor = self.db.aql.execute("FOR d IN drugs RETURN d")

            for drug in tqdm(cursor, desc="Enriching drugs with DrugBank SMILES"):
                self.stats["total_drugs"] += 1

                drug_id = drug.get("drug_id", "")

                # Try to match with DrugBank ID
                if drug_id in drugbank_smiles:
                    smiles = drugbank_smiles[drug_id]

                    update_doc = {"_key": drug["_key"], "smiles": smiles}

                    # Calculate properties
                    properties = self.calculate_properties(smiles)
                    update_doc.update(properties)

                    # Generate embedding
                    embedding = self.generate_embedding(smiles)
                    if embedding:
                        update_doc["embedding"] = embedding
                        self.stats["embeddings_generated"] += 1

                    # Update drug
                    self.drug_collection.update(update_doc, merge=True)
                    self.stats["drugbank_drugs_matched"] += 1
                    self.stats["smiles_added"] += 1
                    self.stats["drugs_enriched"] += 1

            logger.info(
                f"Matched {self.stats['drugbank_drugs_matched']} drugs with DrugBank SMILES"
            )

        except Exception as e:
            logger.error(f"Failed to load DrugBank vocabulary: {e}")

    def generate_report(self):
        """Generate enrichment report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "database": "NeuThera",
            "sources": ["DAVIS", "DrugBank"],
            "enrichment_rate": (
                self.stats["drugs_enriched"] / self.stats["total_drugs"] * 100
            )
            if self.stats["total_drugs"] > 0
            else 0,
        }

        # Save report
        report_file = (
            f"enrichment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("DRUG DATABASE ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Total drugs processed: {self.stats['total_drugs']:,}")
        print(f"Drugs enriched: {self.stats['drugs_enriched']:,}")
        print(f"SMILES added: {self.stats['smiles_added']:,}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']:,}")
        print(f"DAVIS drugs added: {self.stats['davis_drugs_added']:,}")
        print(f"DrugBank matches: {self.stats['drugbank_drugs_matched']:,}")
        print(f"Properties calculated: {self.stats['properties_calculated']:,}")
        print(f"Enrichment rate: {report['enrichment_rate']:.2f}%")
        print("=" * 60 + "\n")

        logger.info(f"Report saved to {report_file}")

        return report


def main():
    """Main execution function"""
    if not LIBRARIES_AVAILABLE:
        print("ERROR: Required libraries not installed")
        sys.exit(1)

    enricher = DrugDataEnricher()

    # Load DAVIS data first (has SMILES)
    enricher.load_davis_data()

    # Then try to match existing drugs with DrugBank vocabulary
    enricher.load_drugbank_vocabulary()

    # Generate report
    report = enricher.generate_report()

    return report


if __name__ == "__main__":
    main()
