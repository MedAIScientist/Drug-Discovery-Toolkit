#!/usr/bin/env python3
"""
Process Real Biological Data for Drug Discovery Toolkit
This script loads real datasets from BioSNAP and populates ArangoDB with a knowledge graph
"""

import os
import sys
import gzip
import json
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import database and chemical libraries
try:
    from arango import ArangoClient
    from dotenv import load_dotenv

    load_dotenv()
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    sys.exit(1)

# Try importing RDKit for chemical processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning(
        "RDKit not available. Chemical property calculations will be skipped."
    )
    RDKIT_AVAILABLE = False


class BiologicalDataProcessor:
    """Process and load real biological datasets into ArangoDB"""

    def __init__(self):
        self.data_dir = Path("./data")
        self.db = None
        self.client = None
        self.stats = defaultdict(int)
        self.init_database()

    def init_database(self):
        """Initialize ArangoDB connection"""
        try:
            # Get credentials from environment or use defaults
            ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
            ARANGO_USER = os.getenv("ARANGO_USER", "root")
            ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

            self.client = ArangoClient(hosts=ARANGO_HOST)

            # Connect to system database
            sys_db = self.client.db(
                "_system", username=ARANGO_USER, password=ARANGO_PASS
            )

            # Create NeuThera database if it doesn't exist
            if not sys_db.has_database("NeuThera"):
                sys_db.create_database("NeuThera")
                logger.info("Created NeuThera database")

            # Connect to NeuThera database
            self.db = self.client.db(
                "NeuThera", username=ARANGO_USER, password=ARANGO_PASS
            )

            # Create collections
            self._create_collections()

            logger.info("Successfully connected to ArangoDB")

        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            logger.info("Continuing without database connection...")
            self.db = None

    def _create_collections(self):
        """Create required collections in ArangoDB"""
        if not self.db:
            return

        # Document collections (nodes)
        node_collections = [
            "drugs",
            "proteins",
            "genes",
            "diseases",
            "functions",
            "tissues",
            "genomic_regions",
        ]

        # Edge collections (relationships)
        edge_collections = [
            "drug_drug",
            "drug_protein",
            "drug_gene",
            "drug_disease",
            "protein_protein",
            "protein_gene",
            "gene_gene",
            "disease_disease",
            "disease_gene",
            "disease_function",
            "function_function",
            "gene_function",
        ]

        # Create node collections
        for collection in node_collections:
            if not self.db.has_collection(collection):
                self.db.create_collection(collection)
                logger.info(f"Created collection: {collection}")

        # Create edge collections
        for collection in edge_collections:
            if not self.db.has_collection(collection):
                self.db.create_collection(collection, edge=True)
                logger.info(f"Created edge collection: {collection}")

    def read_gz_file(self, filepath: Path) -> pd.DataFrame:
        """Read gzipped TSV file"""
        logger.info(f"Reading {filepath.name}...")
        try:
            with gzip.open(filepath, "rt") as f:
                # Try to infer separator
                first_line = f.readline()
                f.seek(0)

                if "\t" in first_line:
                    df = pd.read_csv(f, sep="\t", low_memory=False)
                else:
                    df = pd.read_csv(f, low_memory=False)

                logger.info(f"Loaded {len(df)} rows from {filepath.name}")
                return df

        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return pd.DataFrame()

    def process_drug_drug_interactions(self):
        """Process drug-drug interactions from ChCh-Miner"""
        filepath = self.data_dir / "ChCh-Miner_durgbank-chem-chem.tsv.gz"
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return

        df = self.read_gz_file(filepath)

        if df.empty:
            return

        logger.info("Processing drug-drug interactions...")

        drugs = set()
        interactions = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Drug-Drug"):
            try:
                # Extract drug IDs (assuming columns are drug1, drug2, interaction_type)
                drug1 = str(row.iloc[0])
                drug2 = str(row.iloc[1])

                drugs.add(drug1)
                drugs.add(drug2)

                interaction = {
                    "_from": f"drugs/{drug1}",
                    "_to": f"drugs/{drug2}",
                    "type": "drug_drug_interaction",
                    "source": "ChCh-Miner",
                }

                # Add additional fields if available
                if len(row) > 2:
                    interaction["effect"] = str(row.iloc[2])

                interactions.append(interaction)

            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue

        # Store in database
        self._store_drugs(drugs)
        self._store_interactions(interactions, "drug_drug")

        logger.info(
            f"Processed {len(drugs)} unique drugs and {len(interactions)} interactions"
        )

    def process_drug_gene_interactions(self):
        """Process drug-gene interactions from ChG-Miner"""
        filepath = self.data_dir / "ChG-Miner_miner-chem-gene.tsv.gz"
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return

        df = self.read_gz_file(filepath)

        if df.empty:
            return

        logger.info("Processing drug-gene interactions...")

        drugs = set()
        genes = set()
        interactions = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Drug-Gene"):
            try:
                drug_id = str(row.iloc[0])
                gene_id = str(row.iloc[1])

                drugs.add(drug_id)
                genes.add(gene_id)

                interaction = {
                    "_from": f"drugs/{drug_id}",
                    "_to": f"genes/{gene_id}",
                    "type": "drug_gene_interaction",
                    "source": "ChG-Miner",
                }

                if len(row) > 2:
                    interaction["action"] = str(row.iloc[2])

                interactions.append(interaction)

            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue

        self._store_drugs(drugs)
        self._store_genes(genes)
        self._store_interactions(interactions, "drug_gene")

        logger.info(
            f"Processed {len(drugs)} drugs, {len(genes)} genes, {len(interactions)} interactions"
        )

    def process_disease_drug_interactions(self):
        """Process disease-drug associations from DCh-Miner"""
        filepath = self.data_dir / "DCh-Miner_miner-disease-chemical.tsv.gz"
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return

        df = self.read_gz_file(filepath)

        if df.empty:
            return

        logger.info("Processing disease-drug associations...")

        diseases = set()
        drugs = set()
        interactions = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Disease-Drug"):
            try:
                disease_id = str(row.iloc[0])
                drug_id = str(row.iloc[1])

                diseases.add(disease_id)
                drugs.add(drug_id)

                interaction = {
                    "_from": f"diseases/{disease_id}",
                    "_to": f"drugs/{drug_id}",
                    "type": "disease_drug_association",
                    "source": "DCh-Miner",
                }

                if len(row) > 2:
                    interaction["relationship"] = str(row.iloc[2])

                interactions.append(interaction)

            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue

        self._store_diseases(diseases)
        self._store_drugs(drugs)
        self._store_interactions(interactions, "drug_disease")

        logger.info(
            f"Processed {len(diseases)} diseases, {len(drugs)} drugs, {len(interactions)} associations"
        )

    def process_disease_gene_associations(self):
        """Process disease-gene associations from DG-Miner"""
        filepath = self.data_dir / "DG-Miner_miner-disease-gene.tsv.gz"
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return

        df = self.read_gz_file(filepath)

        if df.empty:
            return

        logger.info("Processing disease-gene associations...")

        diseases = set()
        genes = set()
        interactions = []

        # Process in chunks for large file
        chunk_size = 10000
        total_processed = 0

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]

            for idx, row in chunk.iterrows():
                try:
                    disease_id = str(row.iloc[0])
                    gene_id = str(row.iloc[1])

                    diseases.add(disease_id)
                    genes.add(gene_id)

                    interaction = {
                        "_from": f"diseases/{disease_id}",
                        "_to": f"genes/{gene_id}",
                        "type": "disease_gene_association",
                        "source": "DG-Miner",
                    }

                    if len(row) > 2:
                        interaction["score"] = (
                            float(row.iloc[2]) if pd.notna(row.iloc[2]) else None
                        )

                    interactions.append(interaction)
                    total_processed += 1

                except Exception as e:
                    logger.debug(f"Error processing row {idx}: {e}")
                    continue

            # Store in batches
            if len(interactions) >= 5000:
                self._store_diseases(diseases)
                self._store_genes(genes)
                self._store_interactions(interactions, "disease_gene")
                interactions = []
                diseases = set()
                genes = set()

            logger.info(f"Processed {total_processed} disease-gene associations...")

        # Store remaining
        if interactions:
            self._store_diseases(diseases)
            self._store_genes(genes)
            self._store_interactions(interactions, "disease_gene")

        logger.info(f"Total processed: {total_processed} disease-gene associations")

    def process_gene_function_associations(self):
        """Process gene-function associations from GF-Miner"""
        filepath = self.data_dir / "GF-Miner_miner-gene-function.tsv.gz"
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return

        df = self.read_gz_file(filepath)

        if df.empty:
            return

        logger.info("Processing gene-function associations...")

        genes = set()
        functions = set()
        interactions = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gene-Function"):
            try:
                gene_id = str(row.iloc[0])
                function_id = str(row.iloc[1])

                genes.add(gene_id)
                functions.add(function_id)

                interaction = {
                    "_from": f"genes/{gene_id}",
                    "_to": f"functions/{function_id}",
                    "type": "gene_function_association",
                    "source": "GF-Miner",
                }

                if len(row) > 2:
                    interaction["evidence"] = str(row.iloc[2])

                interactions.append(interaction)

            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue

        self._store_genes(genes)
        self._store_functions(functions)
        self._store_interactions(interactions, "gene_function")

        logger.info(
            f"Processed {len(genes)} genes, {len(functions)} functions, {len(interactions)} associations"
        )

    def process_protein_protein_interactions(self):
        """Process protein-protein interactions if available"""
        # Check for PPI data in existing files or DAVIS dataset
        davis_dir = self.data_dir / "DAVIS"
        if davis_dir.exists():
            logger.info("Processing DAVIS dataset for drug-target interactions...")
            self._process_davis_dataset(davis_dir)

    def _process_davis_dataset(self, davis_dir: Path):
        """Process DAVIS drug-target interaction dataset"""
        try:
            # Read affinity matrix
            affinity_file = davis_dir / "Y.txt"
            if affinity_file.exists():
                Y = np.loadtxt(affinity_file)
                logger.info(f"Loaded affinity matrix with shape {Y.shape}")

            # Read drug SMILES
            drug_file = davis_dir / "drug.txt"
            if drug_file.exists():
                with open(drug_file) as f:
                    drug_smiles = [line.strip() for line in f]
                logger.info(f"Loaded {len(drug_smiles)} drug SMILES")

                # Process and store drugs with SMILES
                for i, smiles in enumerate(drug_smiles):
                    drug_doc = {
                        "_key": f"davis_drug_{i}",
                        "smiles": smiles,
                        "source": "DAVIS",
                    }

                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            drug_doc["molecular_weight"] = Descriptors.ExactMolWt(mol)
                            drug_doc["logp"] = Descriptors.MolLogP(mol)
                            drug_doc["num_rings"] = Chem.rdMolDescriptors.CalcNumRings(
                                mol
                            )

                    self._store_single_drug(drug_doc)

            # Read protein sequences
            protein_file = davis_dir / "target.txt"
            if protein_file.exists():
                with open(protein_file) as f:
                    protein_seqs = [line.strip() for line in f]
                logger.info(f"Loaded {len(protein_seqs)} protein sequences")

                # Process and store proteins
                for i, seq in enumerate(protein_seqs):
                    protein_doc = {
                        "_key": f"davis_protein_{i}",
                        "sequence": seq,
                        "sequence_length": len(seq),
                        "source": "DAVIS",
                    }
                    self._store_single_protein(protein_doc)

            # Store drug-protein interactions with affinity values
            if affinity_file.exists() and drug_file.exists() and protein_file.exists():
                interactions = []
                for i in range(len(drug_smiles)):
                    for j in range(len(protein_seqs)):
                        if i < Y.shape[0] and j < Y.shape[1]:
                            affinity = Y[i, j]
                            interaction = {
                                "_from": f"drugs/davis_drug_{i}",
                                "_to": f"proteins/davis_protein_{j}",
                                "affinity": float(affinity),
                                "type": "drug_target_interaction",
                                "source": "DAVIS",
                            }
                            interactions.append(interaction)

                self._store_interactions(
                    interactions[:1000], "drug_protein"
                )  # Store first 1000 for demo
                logger.info(
                    f"Stored {min(1000, len(interactions))} drug-protein interactions from DAVIS"
                )

        except Exception as e:
            logger.error(f"Error processing DAVIS dataset: {e}")

    def _store_drugs(self, drugs: set):
        """Store drug entities in database"""
        if not self.db or not drugs:
            return

        collection = self.db.collection("drugs")
        documents = []

        for drug_id in drugs:
            doc = {
                "_key": drug_id.replace("/", "_").replace(":", "_"),
                "drug_id": drug_id,
                "source": "BioSNAP",
            }
            documents.append(doc)

        try:
            collection.insert_many(documents, overwrite=True)
            self.stats["drugs"] += len(documents)
        except Exception as e:
            logger.debug(f"Error storing drugs: {e}")

    def _store_single_drug(self, drug_doc: dict):
        """Store a single drug with properties"""
        if not self.db:
            return

        try:
            collection = self.db.collection("drugs")
            collection.insert(drug_doc, overwrite=True)
            self.stats["drugs"] += 1
        except Exception as e:
            logger.debug(f"Error storing drug: {e}")

    def _store_genes(self, genes: set):
        """Store gene entities in database"""
        if not self.db or not genes:
            return

        collection = self.db.collection("genes")
        documents = []

        for gene_id in genes:
            doc = {
                "_key": gene_id.replace("/", "_").replace(":", "_"),
                "gene_id": gene_id,
                "source": "BioSNAP",
            }
            documents.append(doc)

        try:
            collection.insert_many(documents, overwrite=True)
            self.stats["genes"] += len(documents)
        except Exception as e:
            logger.debug(f"Error storing genes: {e}")

    def _store_diseases(self, diseases: set):
        """Store disease entities in database"""
        if not self.db or not diseases:
            return

        collection = self.db.collection("diseases")
        documents = []

        for disease_id in diseases:
            doc = {
                "_key": disease_id.replace("/", "_").replace(":", "_"),
                "disease_id": disease_id,
                "source": "BioSNAP",
            }
            documents.append(doc)

        try:
            collection.insert_many(documents, overwrite=True)
            self.stats["diseases"] += len(documents)
        except Exception as e:
            logger.debug(f"Error storing diseases: {e}")

    def _store_functions(self, functions: set):
        """Store function entities in database"""
        if not self.db or not functions:
            return

        collection = self.db.collection("functions")
        documents = []

        for function_id in functions:
            doc = {
                "_key": function_id.replace("/", "_").replace(":", "_"),
                "function_id": function_id,
                "source": "BioSNAP",
            }
            documents.append(doc)

        try:
            collection.insert_many(documents, overwrite=True)
            self.stats["functions"] += len(documents)
        except Exception as e:
            logger.debug(f"Error storing functions: {e}")

    def _store_single_protein(self, protein_doc: dict):
        """Store a single protein with properties"""
        if not self.db:
            return

        try:
            collection = self.db.collection("proteins")
            collection.insert(protein_doc, overwrite=True)
            self.stats["proteins"] += 1
        except Exception as e:
            logger.debug(f"Error storing protein: {e}")

    def _store_interactions(self, interactions: list, collection_name: str):
        """Store interaction edges in database"""
        if not self.db or not interactions:
            return

        collection = self.db.collection(collection_name)

        # Process in batches
        batch_size = 1000
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i : i + batch_size]
            try:
                collection.insert_many(batch, overwrite=True)
                self.stats[collection_name] += len(batch)
            except Exception as e:
                logger.debug(f"Error storing interactions in {collection_name}: {e}")

    def process_all_datasets(self):
        """Process all available datasets"""
        logger.info("=" * 60)
        logger.info("Starting processing of real biological datasets")
        logger.info("=" * 60)

        # Process each dataset type
        self.process_drug_drug_interactions()
        self.process_drug_gene_interactions()
        self.process_disease_drug_interactions()
        self.process_disease_gene_associations()
        self.process_gene_function_associations()
        self.process_protein_protein_interactions()

        # Print summary statistics
        logger.info("=" * 60)
        logger.info("Processing complete! Summary:")
        logger.info("-" * 60)
        for entity_type, count in self.stats.items():
            logger.info(f"{entity_type}: {count:,}")
        logger.info("=" * 60)

        return self.stats


def main():
    """Main execution function"""
    processor = BiologicalDataProcessor()
    stats = processor.process_all_datasets()

    # Save summary to file
    summary_file = Path("data_processing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "data_source": "BioSNAP and DAVIS datasets",
                "database": "ArangoDB (NeuThera)",
            },
            f,
            indent=2,
        )

    logger.info(f"Summary saved to {summary_file}")

    return stats


if __name__ == "__main__":
    main()
