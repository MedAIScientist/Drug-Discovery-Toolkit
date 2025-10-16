"""
Updated tools module with hybrid model integration for proper handling of mixed feature types.
This module properly separates sequence-based and structural features.
"""

import json
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing dependencies
try:
    import streamlit as st
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, MACCSkeys, DataStructs
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    from Bio.PDB import MMCIFParser
    import plotly.graph_objects as go
    from pyvis.network import Network
    from dotenv import load_dotenv
    from arango import ArangoClient
    from langchain_community.graphs import ArangoGraph
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain.tools import Tool
    from langchain.callbacks.base import BaseCallbackHandler
    from DeepPurpose import utils, DTI as models
except ImportError as e:
    logger.warning(f"Some dependencies not available: {e}")

# Import hybrid model components
from hybrid_model import (
    HybridDrugDiscoveryModel,
    FeatureConfig,
    FeatureValidator,
    HybridPredictor,
)

# Load environment variables
load_dotenv()

# ================= Configuration =================


@dataclass
class ModelConfig:
    """Configuration for model selection and feature handling"""

    use_hybrid_model: bool = False
    sequence_models: Dict[str, str] = None
    max_sequence_length: int = 512
    enable_structural_features: bool = False

    def __post_init__(self):
        if self.sequence_models is None:
            self.sequence_models = {
                "drug": "seyonec/ChemBERTa-zinc-base-v1",
                "protein": "Rostlab/prot_bert",
                "dna": "zhihan1996/DNABERT-2-117M",
            }


# ================= Database Connection =================


def initialize_database():
    """Initialize ArangoDB connection with proper error handling"""
    try:
        ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
        ARANGO_USER = os.getenv("ARANGO_USER", "root")
        ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

        client = ArangoClient(hosts=ARANGO_HOST)
        sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASS)

        if not sys_db.has_database("NeuThera"):
            sys_db.create_database("NeuThera")
            logger.info("Created NeuThera database")

        db = client.db("NeuThera", username=ARANGO_USER, password=ARANGO_PASS)

        # Create collections if needed
        required_collections = ["drugs", "proteins", "drug_protein_links"]
        for collection_name in required_collections:
            if not db.has_collection(collection_name):
                if collection_name == "drug_protein_links":
                    db.create_collection(collection_name, edge=True)
                else:
                    db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")

        arango_graph = ArangoGraph(db)
        return db, arango_graph
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return None, None


# ================= Feature Processing =================


class FeatureProcessor:
    """Processes and validates different types of features"""

    @staticmethod
    def classify_features(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Classify input features by type"""
        sequence_features = []
        tabular_features = []
        categorical_features = []

        for key, value in input_data.items():
            if key in ["drug_smiles", "x_drug", "smiles"]:
                sequence_features.append("drug_smiles")
            elif key in ["protein_sequence", "x_target", "target_sequence"]:
                sequence_features.append("protein_sequence")
            elif key in ["dna_sequence", "genomic_sequence"]:
                sequence_features.append("dna_sequence")
            elif key in [
                "distance_to_stop_codon",
                "expression_level",
                "gc_content",
                "molecular_weight",
                "logp",
                "binding_site_depth",
            ]:
                tabular_features.append(key)
            elif key in ["gene_id", "chromosome", "strand", "organism"]:
                categorical_features.append(key)

        return {
            "sequence": sequence_features,
            "tabular": tabular_features,
            "categorical": categorical_features,
        }

    @staticmethod
    def prepare_features(
        input_data: Dict[str, Any], feature_types: Dict[str, List[str]]
    ) -> Dict:
        """Prepare features for model input"""
        prepared = {"sequence_data": {}, "tabular_data": None, "categorical_data": {}}

        # Process sequence features
        for feat in feature_types["sequence"]:
            if feat == "drug_smiles":
                value = (
                    input_data.get("drug_smiles")
                    or input_data.get("x_drug")
                    or input_data.get("smiles")
                )
                if value:
                    prepared["sequence_data"]["drug_smiles"] = [value]
            elif feat == "protein_sequence":
                value = (
                    input_data.get("protein_sequence")
                    or input_data.get("x_target")
                    or input_data.get("target_sequence")
                )
                if value:
                    prepared["sequence_data"]["protein_sequence"] = [value]
            elif feat == "dna_sequence":
                value = input_data.get("dna_sequence") or input_data.get(
                    "genomic_sequence"
                )
                if value:
                    prepared["sequence_data"]["dna_sequence"] = [value]

        # Process tabular features
        if feature_types["tabular"]:
            tabular_values = []
            for feat in feature_types["tabular"]:
                if feat in input_data:
                    try:
                        tabular_values.append(float(input_data[feat]))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {feat} to float, using 0.0")
                        tabular_values.append(0.0)

            if tabular_values:
                prepared["tabular_data"] = torch.tensor(
                    [tabular_values], dtype=torch.float32
                )

        # Process categorical features
        for feat in feature_types["categorical"]:
            if feat in input_data:
                # Note: In production, you'd need proper encoding/vocabulary mapping
                prepared["categorical_data"][feat] = torch.tensor(
                    [hash(input_data[feat]) % 10000]
                )

        return prepared


# ================= Enhanced Prediction Functions =================


def PredictBindingAffinityHybrid(
    input_data: Union[str, Dict],
    use_structural_features: bool = True,
    model_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """
    Enhanced binding affinity prediction with proper feature handling

    Args:
        input_data: Input features as JSON string or dictionary
        use_structural_features: Whether to use structural features if available
        model_config: Model configuration (uses defaults if None)

    Returns:
        Dictionary containing prediction and metadata
    """

    if model_config is None:
        model_config = ModelConfig()

    # Parse input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    # Classify features
    feature_processor = FeatureProcessor()
    feature_types = feature_processor.classify_features(input_data)

    # Log feature analysis
    logger.info(f"Detected feature types: {feature_types}")

    # Validate feature-model compatibility
    validator = FeatureValidator()
    has_mixed_features = len(feature_types["sequence"]) > 0 and (
        len(feature_types["tabular"]) > 0 or len(feature_types["categorical"]) > 0
    )

    result = {
        "prediction": None,
        "model_used": None,
        "features_used": feature_types,
        "warnings": [],
    }

    try:
        if has_mixed_features and use_structural_features:
            # Use hybrid model
            logger.info("Using hybrid model for mixed feature types")

            # Create feature configuration
            feature_config = FeatureConfig(
                sequence_features=feature_types["sequence"],
                tabular_features=feature_types["tabular"],
                categorical_features=feature_types["categorical"],
            )

            # Initialize hybrid model
            model = HybridDrugDiscoveryModel(
                feature_config=feature_config, task="binding_affinity"
            )

            # Prepare features
            prepared_features = feature_processor.prepare_features(
                input_data, feature_types
            )

            # Make prediction
            with torch.no_grad():
                output = model(
                    sequence_data=prepared_features["sequence_data"],
                    tabular_data=prepared_features["tabular_data"],
                    categorical_data=prepared_features["categorical_data"],
                )
                prediction = output.item()

            result["prediction"] = prediction
            result["model_used"] = "hybrid"

        else:
            # Use sequence-only model
            logger.info("Using sequence-only model")

            # Extract sequence data
            drug_smiles = (
                input_data.get("x_drug")
                or input_data.get("drug_smiles")
                or input_data.get("smiles")
            )
            protein_sequence = (
                input_data.get("x_target")
                or input_data.get("protein_sequence")
                or input_data.get("target_sequence")
            )

            if not drug_smiles or not protein_sequence:
                raise ValueError("Both drug SMILES and protein sequence are required")

            # Check for ignored features
            if has_mixed_features and not use_structural_features:
                result["warnings"].append(
                    "Structural features detected but ignored. "
                    "Set use_structural_features=True to use them."
                )

            # Use DeepPurpose for pure sequence prediction
            binding_model = models.model_pretrained(path_dir="DTI_model")
            X_pred = utils.data_process(
                [drug_smiles],
                [protein_sequence],
                [7.635],
                drug_encoding="CNN",
                target_encoding="CNN",
                split_method="no_split",
            )
            predictions = binding_model.predict(X_pred)

            result["prediction"] = float(predictions[0])
            result["model_used"] = "DeepPurpose_CNN"

        # Add interpretation
        result["interpretation"] = interpret_binding_affinity(result["prediction"])

        # Display results if in Streamlit
        if "streamlit" in sys.modules:
            display_prediction_results(result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        result["error"] = str(e)
        result["prediction"] = 0.0

    return result


def interpret_binding_affinity(affinity: float) -> Dict[str, str]:
    """Interpret binding affinity value"""
    interpretation = {
        "value": f"{affinity:.3f}",
        "unit": "log(Kd/Ki)",
        "strength": "",
        "description": "",
    }

    if affinity < 5:
        interpretation["strength"] = "Weak"
        interpretation["description"] = "Minimal binding expected (mM range)"
    elif 5 <= affinity < 7:
        interpretation["strength"] = "Moderate"
        interpretation["description"] = "Moderate binding (μM range)"
    elif 7 <= affinity < 9:
        interpretation["strength"] = "Strong"
        interpretation["description"] = "Strong binding (nM range)"
    else:
        interpretation["strength"] = "Very Strong"
        interpretation["description"] = "Very strong binding (pM range)"

    return interpretation


def display_prediction_results(result: Dict[str, Any]):
    """Display prediction results in Streamlit sidebar"""
    try:
        if st.sidebar:
            with st.sidebar:
                st.markdown("### 🔬 Binding Affinity Prediction")

                if "error" in result:
                    st.error(f"❌ Prediction failed: {result['error']}")
                else:
                    # Display prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Affinity", f"{result['prediction']:.3f}")
                    with col2:
                        st.metric("Model Used", result["model_used"])

                    # Display interpretation
                    if "interpretation" in result:
                        interp = result["interpretation"]
                        st.info(
                            f"**{interp['strength']} Binding**\n{interp['description']}"
                        )

                    # Display feature breakdown
                    with st.expander("📊 Features Used"):
                        for feat_type, features in result["features_used"].items():
                            if features:
                                st.write(
                                    f"**{feat_type.capitalize()}:** {', '.join(features)}"
                                )

                    # Display warnings
                    if result.get("warnings"):
                        with st.expander("⚠️ Warnings"):
                            for warning in result["warnings"]:
                                st.warning(warning)

                st.divider()
    except Exception as e:
        logger.error(f"Failed to display results: {e}")


# ================= Feature Validation Tool =================


def ValidateFeatures(
    features: Dict[str, str], model_type: str = "auto"
) -> Dict[str, Any]:
    """
    Validate feature-model compatibility

    Args:
        features: Dictionary mapping feature names to their types
        model_type: Type of model ('auto', 'sequence_transformer', 'hybrid', etc.)

    Returns:
        Validation result with recommendations
    """
    validator = FeatureValidator()

    if model_type == "auto":
        # Auto-detect best model type
        has_sequences = any(v in ["sequence", "text"] for v in features.values())
        has_tabular = any(
            v in ["continuous", "categorical_encoded"] for v in features.values()
        )

        if has_sequences and has_tabular:
            model_type = "hybrid"
        elif has_sequences:
            model_type = "sequence_transformer"
        else:
            model_type = "tabular_nn"

    # Validate based on model type
    if model_type == "hybrid":
        # Hybrid models can handle everything
        is_valid = True
        issues = []
        recommendation = "Hybrid model can handle all feature types"
    else:
        is_valid, issues = validator.validate(features, model_type)
        if is_valid:
            recommendation = f"{model_type} is appropriate for these features"
        else:
            recommendation = "Consider using a hybrid model for mixed feature types"

    return {
        "is_valid": is_valid,
        "model_type": model_type,
        "issues": issues,
        "recommendation": recommendation,
        "feature_summary": {
            "total_features": len(features),
            "feature_types": list(set(features.values())),
        },
    }


# ================= Tool Definitions =================

# Initialize database and models
db, arango_graph = initialize_database()

# Create enhanced tools with proper feature handling
predict_binding_affinity_hybrid = Tool(
    name="PredictBindingAffinityHybrid",
    func=PredictBindingAffinityHybrid,
    description="""
    AI-POWERED BINDING AFFINITY PREDICTION with proper feature handling.

    Automatically detects and properly processes:
    - Sequence features (SMILES, protein sequences, DNA)
    - Structural features (distances, expression levels, genomic coordinates)
    - Categorical features (gene IDs, chromosomes)

    Uses appropriate model architecture based on input features.
    """,
)

validate_features_tool = Tool(
    name="ValidateFeatures",
    func=ValidateFeatures,
    description="""
    FEATURE-MODEL COMPATIBILITY VALIDATOR

    Ensures features are compatible with the selected model architecture.
    Prevents mixing incompatible feature types with sequence transformers.
    """,
)

# Export enhanced tools
enhanced_tools = [predict_binding_affinity_hybrid, validate_features_tool]

# ================= Testing =================

if __name__ == "__main__":
    # Test 1: Sequence-only prediction
    print("Test 1: Sequence-only prediction")
    result1 = PredictBindingAffinityHybrid(
        {
            "x_drug": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1",
            "x_target": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW",
        },
        use_structural_features=False,
    )
    print(f"Result: {result1}\n")

    # Test 2: Mixed features with hybrid model
    print("Test 2: Mixed features with hybrid model")
    result2 = PredictBindingAffinityHybrid(
        {
            "x_drug": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1",
            "x_target": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW",
            "gene_id": "ENSG00000135218",
            "distance_to_stop_codon": 1500,
            "expression_level": 2.5,
        },
        use_structural_features=True,
    )
    print(f"Result: {result2}\n")

    # Test 3: Feature validation
    print("Test 3: Feature validation")
    validation_result = ValidateFeatures(
        {
            "smiles": "sequence",
            "gene_id": "categorical_raw",
            "expression": "continuous",
        },
        model_type="auto",
    )
    print(f"Validation: {validation_result}\n")

    print("All tests completed!")
