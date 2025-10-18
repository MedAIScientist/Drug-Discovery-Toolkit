#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Drug Discovery Toolkit
Implements all analyses from start.ipynb using real biological data
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    # Database and environment
    from arango import ArangoClient
    from dotenv import load_dotenv

    load_dotenv()

    # Machine learning and embeddings
    from transformers import AutoTokenizer, AutoModel
    import torch
    import faiss

    # Chemical informatics
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys, AllChem, Descriptors, Draw

    # Protein structure
    from Bio.PDB import MMCIFParser

    # Drug-Target Interaction
    from DeepPurpose import utils, DTI as models

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Graph processing
    from langchain_community.graphs import ArangoGraph
    from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain_google_genai import ChatGoogleGenerativeAI

    LIBRARIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    LIBRARIES_AVAILABLE = False


class DrugDiscoveryAnalysis:
    """Comprehensive analysis system for drug discovery using real biological data"""

    def __init__(self):
        """Initialize the analysis system"""
        self.db = None
        self.arango_graph = None
        self.drug_collection = None
        self.tokenizer = None
        self.model = None
        self.faiss_index = None
        self.drug_keys = []
        self.embeddings = []

        # Initialize components
        self._init_database()
        self._init_models()
        self._load_embeddings()

    def _init_database(self):
        """Initialize ArangoDB connection"""
        try:
            ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
            ARANGO_USER = os.getenv("ARANGO_USER", "root")
            ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

            client = ArangoClient(hosts=ARANGO_HOST)
            self.db = client.db("NeuThera", username=ARANGO_USER, password=ARANGO_PASS)

            # Initialize graph
            if self.db.has_graph("NeuThera"):
                self.arango_graph = ArangoGraph(self.db)
                self.drug_collection = self.db.collection("drugs")
                logger.info("Successfully connected to NeuThera database")
            else:
                logger.warning("NeuThera graph not found in database")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")

    def _init_models(self):
        """Initialize ML models"""
        try:
            # Initialize ChemBERTa for molecular embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(
                "seyonec/ChemBERTa-zinc-base-v1"
            )
            self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            logger.info("ChemBERTa model loaded successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")

    def _load_embeddings(self):
        """Load drug embeddings from database and create FAISS index"""
        if not self.db:
            return

        try:
            # Query all drug embeddings
            cursor = self.db.aql.execute(
                "FOR doc IN drugs FILTER doc.embedding != null "
                "RETURN {key: doc._key, embedding: doc.embedding, smiles: doc.smiles}"
            )

            self.drug_keys = []
            self.embeddings = []
            smiles_list = []

            for doc in cursor:
                if doc and "embedding" in doc and doc["embedding"]:
                    self.drug_keys.append(doc["key"])
                    self.embeddings.append(doc["embedding"])
                    smiles_list.append(doc.get("smiles", ""))

            if self.embeddings:
                # Convert to numpy array
                self.embeddings = np.array(self.embeddings, dtype=np.float32)

                # Create FAISS index for similarity search
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(self.embeddings)

                logger.info(
                    f"Loaded {len(self.embeddings)} drug embeddings (dimension: {dimension})"
                )
                logger.info(f"FAISS index created for similarity search")
            else:
                logger.warning("No embeddings found in database")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")

    def get_chemberta_embedding(self, smiles: str) -> Optional[List[float]]:
        """Generate ChemBERTa embedding for a molecule"""
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        try:
            inputs = self.tokenizer(
                smiles, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed for {smiles}: {e}")
            return None

    def analyze_database_statistics(self) -> Dict[str, Any]:
        """Analyze statistics of the loaded biological data"""
        stats = {}

        if not self.db:
            return {"error": "Database not connected"}

        try:
            # Count entities in each collection
            collections_to_analyze = [
                "drugs",
                "proteins",
                "genes",
                "diseases",
                "functions",
            ]

            for collection_name in collections_to_analyze:
                if self.db.has_collection(collection_name):
                    count = self.db.collection(collection_name).count()
                    stats[f"{collection_name}_count"] = count

            # Count relationships
            edge_collections = [
                "drug_drug",
                "drug_protein",
                "drug_gene",
                "drug_disease",
                "protein_protein",
                "gene_gene",
                "disease_disease",
            ]

            for edge_name in edge_collections:
                if self.db.has_collection(edge_name):
                    count = self.db.collection(edge_name).count()
                    stats[f"{edge_name}_edges"] = count

            # Analyze drug properties
            drug_stats = self._analyze_drug_properties()
            stats["drug_properties"] = drug_stats

            # Network statistics
            network_stats = self._analyze_network_statistics()
            stats["network"] = network_stats

            return stats

        except Exception as e:
            logger.error(f"Statistics analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_drug_properties(self) -> Dict[str, Any]:
        """Analyze chemical properties of drugs"""
        properties = {
            "with_smiles": 0,
            "with_embeddings": 0,
            "molecular_weight_stats": {},
            "logp_stats": {},
        }

        try:
            # Count drugs with SMILES
            cursor = self.db.aql.execute(
                "FOR d IN drugs FILTER d.smiles != null "
                "RETURN {mw: d.molecular_weight, logp: d.logp}"
            )

            mw_values = []
            logp_values = []

            for doc in cursor:
                properties["with_smiles"] += 1
                if doc.get("mw"):
                    mw_values.append(doc["mw"])
                if doc.get("logp"):
                    logp_values.append(doc["logp"])

            # Calculate statistics
            if mw_values:
                properties["molecular_weight_stats"] = {
                    "mean": np.mean(mw_values),
                    "std": np.std(mw_values),
                    "min": np.min(mw_values),
                    "max": np.max(mw_values),
                }

            if logp_values:
                properties["logp_stats"] = {
                    "mean": np.mean(logp_values),
                    "std": np.std(logp_values),
                    "min": np.min(logp_values),
                    "max": np.max(logp_values),
                }

            properties["with_embeddings"] = len(self.embeddings)

        except Exception as e:
            logger.error(f"Drug property analysis failed: {e}")

        return properties

    def _analyze_network_statistics(self) -> Dict[str, Any]:
        """Analyze network topology statistics"""
        stats = {}

        try:
            # Calculate node degrees
            query = """
            FOR d IN drugs
                LET outgoing = LENGTH(FOR v IN 1..1 OUTBOUND d GRAPH 'NeuThera' RETURN v)
                LET incoming = LENGTH(FOR v IN 1..1 INBOUND d GRAPH 'NeuThera' RETURN v)
                RETURN {out: outgoing, in: incoming, total: outgoing + incoming}
            """

            cursor = self.db.aql.execute(query)
            degrees = [doc["total"] for doc in cursor if doc]

            if degrees:
                stats["average_degree"] = np.mean(degrees)
                stats["max_degree"] = np.max(degrees)
                stats["min_degree"] = np.min(degrees)
                stats["degree_std"] = np.std(degrees)

        except Exception as e:
            logger.debug(f"Network statistics failed: {e}")

        return stats

    def find_similar_drugs(
        self, query_smiles: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar drugs using FAISS similarity search"""
        if not self.faiss_index:
            logger.error("FAISS index not initialized")
            return []

        try:
            # Get embedding for query molecule
            query_embedding = self.get_chemberta_embedding(query_smiles)
            if not query_embedding:
                return []

            # Search using FAISS
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_vector, top_k)

            # Get drug information
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.drug_keys):
                    drug_key = self.drug_keys[idx]

                    # Get additional information from database
                    drug_doc = self.drug_collection.get(drug_key)

                    results.append(
                        {
                            "drug_id": drug_key,
                            "similarity_score": 1
                            / (1 + distance),  # Convert distance to similarity
                            "smiles": drug_doc.get("smiles", ""),
                            "source": drug_doc.get("source", ""),
                            "distance": float(distance),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def predict_binding_affinity(self, drug_smiles: str, target_sequence: str) -> float:
        """Predict drug-target binding affinity using DeepPurpose"""
        try:
            X_drug = [drug_smiles]
            X_target = [target_sequence]
            y = [7.635]  # Default value for prediction

            # Load pre-trained model
            model = models.model_pretrained(path_dir="./DTI_model")

            # Process data
            X_pred = utils.data_process(
                X_drug,
                X_target,
                y,
                drug_encoding="CNN",
                target_encoding="CNN",
                split_method="no_split",
            )

            # Make prediction
            predictions = model.predict(X_pred)
            return float(predictions[0])

        except Exception as e:
            logger.error(f"Binding affinity prediction failed: {e}")
            return 0.0

    def cluster_drugs(self, n_clusters: int = 10) -> Dict[str, Any]:
        """Perform clustering analysis on drug embeddings"""
        if len(self.embeddings) == 0:
            return {"error": "No embeddings available"}

        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)

            # Calculate silhouette score
            silhouette = silhouette_score(self.embeddings, cluster_labels)

            # Get cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_size = np.sum(cluster_mask)
                cluster_stats[f"cluster_{i}"] = {
                    "size": int(cluster_size),
                    "percentage": float(cluster_size / len(cluster_labels) * 100),
                }

            # Store cluster assignments in database
            for drug_key, cluster_id in zip(self.drug_keys, cluster_labels):
                try:
                    self.drug_collection.update(
                        {"_key": drug_key}, {"cluster_id": int(cluster_id)}
                    )
                except:
                    pass

            return {
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette),
                "cluster_statistics": cluster_stats,
                "total_drugs": len(self.embeddings),
            }

        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return {"error": str(e)}

    def visualize_drug_space(
        self, method: str = "pca", n_components: int = 2
    ) -> Dict[str, Any]:
        """Visualize drug chemical space using dimensionality reduction"""
        if len(self.embeddings) == 0:
            return {"error": "No embeddings available"}

        try:
            # Perform dimensionality reduction
            if method.lower() == "pca":
                reducer = PCA(n_components=n_components)
                reduced_embeddings = reducer.fit_transform(self.embeddings)
                variance_explained = reducer.explained_variance_ratio_

                result = {
                    "method": "PCA",
                    "variance_explained": variance_explained.tolist(),
                    "total_variance": float(np.sum(variance_explained)),
                }

            elif method.lower() == "tsne":
                reducer = TSNE(n_components=n_components, random_state=42)
                reduced_embeddings = reducer.fit_transform(
                    self.embeddings[:1000]
                )  # Limit for performance

                result = {
                    "method": "t-SNE",
                    "n_samples": min(1000, len(self.embeddings)),
                }
            else:
                return {"error": f"Unknown method: {method}"}

            # Create visualization
            plt.figure(figsize=(10, 8))

            if n_components == 2:
                plt.scatter(
                    reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, s=10
                )
                plt.xlabel(f"{method} Component 1")
                plt.ylabel(f"{method} Component 2")
                plt.title(f"Drug Chemical Space Visualization ({method})")

                # Save plot
                plot_path = f"drug_space_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                result["plot_saved"] = plot_path
                result["embedding_shape"] = reduced_embeddings.shape

            return result

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return {"error": str(e)}

    def analyze_drug_drug_interactions(self) -> Dict[str, Any]:
        """Analyze drug-drug interaction network"""
        if not self.db:
            return {"error": "Database not connected"}

        try:
            # Get interaction statistics
            query = """
            FOR edge IN drug_drug
                COLLECT WITH COUNT INTO length
                RETURN length
            """
            cursor = self.db.aql.execute(query)
            total_interactions = list(cursor)[0] if cursor else 0

            # Find most connected drugs
            query = """
            FOR d IN drugs
                LET interactions = LENGTH(
                    FOR v IN 1..1 ANY d drug_drug
                    RETURN v
                )
                SORT interactions DESC
                LIMIT 10
                RETURN {drug: d._key, interaction_count: interactions}
            """

            cursor = self.db.aql.execute(query)
            top_drugs = list(cursor)

            # Calculate network density
            n_drugs = self.db.collection("drugs").count()
            max_possible_edges = n_drugs * (n_drugs - 1) / 2
            density = (
                total_interactions / max_possible_edges if max_possible_edges > 0 else 0
            )

            return {
                "total_interactions": total_interactions,
                "network_density": float(density),
                "top_interacting_drugs": top_drugs,
                "total_drugs": n_drugs,
            }

        except Exception as e:
            logger.error(f"Drug interaction analysis failed: {e}")
            return {"error": str(e)}

    def analyze_disease_drug_associations(self) -> Dict[str, Any]:
        """Analyze disease-drug associations"""
        if not self.db:
            return {"error": "Database not connected"}

        try:
            # Get top diseases by drug count
            query = """
            FOR d IN diseases
                LET drug_count = LENGTH(
                    FOR v IN 1..1 OUTBOUND d drug_disease
                    RETURN v
                )
                FILTER drug_count > 0
                SORT drug_count DESC
                LIMIT 20
                RETURN {
                    disease: d.disease_id,
                    associated_drugs: drug_count
                }
            """

            cursor = self.db.aql.execute(query)
            top_diseases = list(cursor)

            # Get drug repurposing opportunities (drugs associated with multiple diseases)
            query = """
            FOR drug IN drugs
                LET disease_count = LENGTH(
                    FOR d IN 1..1 INBOUND drug drug_disease
                    RETURN DISTINCT d
                )
                FILTER disease_count > 1
                SORT disease_count DESC
                LIMIT 20
                RETURN {
                    drug: drug._key,
                    disease_associations: disease_count
                }
            """

            cursor = self.db.aql.execute(query)
            repurposing_candidates = list(cursor)

            return {
                "top_diseases_by_drug_count": top_diseases,
                "drug_repurposing_candidates": repurposing_candidates,
                "total_associations": self.db.collection("drug_disease").count()
                if self.db.has_collection("drug_disease")
                else 0,
            }

        except Exception as e:
            logger.error(f"Disease-drug analysis failed: {e}")
            return {"error": str(e)}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        logger.info("Generating comprehensive analysis report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "database_statistics": self.analyze_database_statistics(),
            "drug_drug_interactions": self.analyze_drug_drug_interactions(),
            "disease_drug_associations": self.analyze_disease_drug_associations(),
            "clustering_analysis": self.cluster_drugs(n_clusters=10),
            "embedding_statistics": {
                "total_embeddings": len(self.embeddings),
                "embedding_dimension": self.embeddings.shape[1]
                if len(self.embeddings) > 0
                else 0,
                "drugs_with_embeddings": len(self.drug_keys),
            },
        }

        # Visualizations
        logger.info("Creating visualizations...")
        report["visualizations"] = {
            "pca": self.visualize_drug_space(method="pca"),
            "tsne": self.visualize_drug_space(method="tsne"),
        }

        # Save report to file
        report_file = f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {report_file}")

        return report

    def run_example_queries(self):
        """Run example analyses to demonstrate functionality"""
        logger.info("Running example analyses...")

        examples = {}

        # Example 1: Find similar drugs
        if self.drug_keys:
            # Get a sample drug
            sample_drug = self.drug_collection.get(self.drug_keys[0])
            if sample_drug and sample_drug.get("smiles"):
                similar_drugs = self.find_similar_drugs(sample_drug["smiles"], top_k=5)
                examples["similar_drugs"] = {
                    "query_drug": self.drug_keys[0],
                    "similar_drugs": similar_drugs,
                }
                logger.info(f"Found {len(similar_drugs)} similar drugs")

        # Example 2: Predict binding affinity (with dummy data for demo)
        try:
            example_smiles = "CC(=O)NC1=CC=C(C=C1)O"  # Paracetamol
            example_sequence = "MKVLWAALLVTFLAGCQAKVEQAVE"  # Sample protein sequence
            affinity = self.predict_binding_affinity(example_smiles, example_sequence)
            examples["binding_affinity_prediction"] = {
                "drug_smiles": example_smiles,
                "target_sequence": example_sequence[:20] + "...",
                "predicted_affinity": affinity,
            }
        except:
            logger.warning("Binding affinity prediction example failed")

        return examples


def main():
    """Main execution function"""
    # Initialize analysis system
    analyzer = DrugDiscoveryAnalysis()

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    # Run example queries
    examples = analyzer.run_example_queries()
    report["example_analyses"] = examples

    # Print summary
    print("\n" + "=" * 60)
    print("DRUG DISCOVERY TOOLKIT - COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 60)

    if "database_statistics" in report:
        stats = report["database_statistics"]
        print("\nDatabase Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(
                    f"  {key}: {value:,}"
                    if isinstance(value, int)
                    else f"  {key}: {value}"
                )

    if "clustering_analysis" in report:
        cluster = report["clustering_analysis"]
        print("\nClustering Analysis:")
        print("-" * 40)
        print(f"  Number of clusters: {cluster.get('n_clusters', 'N/A')}")
        print(
            f"  Silhouette score: {cluster.get('silhouette_score', 'N/A'):.3f}"
            if "silhouette_score" in cluster
            else ""
        )

    if "drug_drug_interactions" in report:
        ddi = report["drug_drug_interactions"]
        print("\nDrug-Drug Interactions:")
        print("-" * 40)
        print(f"  Total interactions: {ddi.get('total_interactions', 0):,}")
        print(f"  Network density: {ddi.get('network_density', 0):.6f}")

    print("\n" + "=" * 60)
    print(f"Full report saved to: comprehensive_analysis_report_*.json")
    print("Visualization plots saved to: drug_space_*.png")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    # Check if all required libraries are available
    if not LIBRARIES_AVAILABLE:
        print("ERROR: Required libraries are not installed.")
        print("Please install: pip install -r requirements.txt")
        sys.exit(1)

    # Run analysis
    main()
