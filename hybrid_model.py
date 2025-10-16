"""
Hybrid Model Architecture for Drug Discovery
Properly handles mixed feature types (sequences, tabular, structural)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for different feature types"""

    sequence_features: List[str] = None
    tabular_features: List[str] = None
    categorical_features: List[str] = None

    def __post_init__(self):
        self.sequence_features = self.sequence_features or []
        self.tabular_features = self.tabular_features or []
        self.categorical_features = self.categorical_features or []


class SequenceEncoder(nn.Module):
    """Encoder for sequence data (SMILES, proteins, DNA)"""

    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences to embeddings"""
        inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings


class TabularEncoder(nn.Module):
    """Encoder for tabular/structural features"""

    def __init__(
        self, input_dim: int, hidden_dims: List[int] = None, output_dim: int = 128
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, tabular_features: torch.Tensor) -> torch.Tensor:
        """Encode tabular features"""
        return self.encoder(tabular_features)


class CategoricalEncoder(nn.Module):
    """Encoder for categorical features (gene IDs, chromosome locations, etc.)"""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int = 32):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(vocab_size, embedding_dim)
                for feature_name, vocab_size in vocab_sizes.items()
            }
        )
        self.output_dim = embedding_dim * len(vocab_sizes)

    def forward(self, categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode categorical features"""
        embeddings = []
        for feature_name, indices in categorical_features.items():
            if feature_name in self.embeddings:
                embeddings.append(self.embeddings[feature_name](indices))

        if embeddings:
            return torch.cat(embeddings, dim=-1)
        else:
            return torch.zeros(1, self.output_dim)


class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple modalities"""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()

        # Project each modality to same dimension
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for input_dim in input_dims]
        )

        # Attention mechanism
        self.attention_query = nn.Linear(output_dim, output_dim)
        self.attention_key = nn.Linear(output_dim, output_dim)
        self.attention_value = nn.Linear(output_dim, output_dim)

        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple embeddings using attention"""

        # Project all embeddings to same dimension
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]

        # Stack for attention computation
        stacked = torch.stack(projected, dim=1)  # [batch_size, num_modalities, dim]

        # Compute attention
        Q = self.attention_query(stacked)
        K = self.attention_key(stacked)
        V = self.attention_value(stacked)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Aggregate across modalities
        fused = attended.mean(dim=1)

        return self.output_projection(fused)


class HybridDrugDiscoveryModel(nn.Module):
    """
    Hybrid model that properly handles different feature types
    for drug discovery tasks
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        task: str = "binding_affinity",
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        self.feature_config = feature_config
        self.task = task
        self.encoders = nn.ModuleDict()
        self.embedding_dims = []

        # Initialize sequence encoders
        if "drug_smiles" in feature_config.sequence_features:
            self.encoders["drug_smiles"] = SequenceEncoder(
                "seyonec/ChemBERTa-zinc-base-v1"
            )
            self.embedding_dims.append(768)  # BERT hidden size

        if "protein_sequence" in feature_config.sequence_features:
            self.encoders["protein_sequence"] = SequenceEncoder("Rostlab/prot_bert")
            self.embedding_dims.append(768)

        if "dna_sequence" in feature_config.sequence_features:
            self.encoders["dna_sequence"] = SequenceEncoder("zhihan1996/DNABERT-2-117M")
            self.embedding_dims.append(768)

        # Initialize tabular encoder
        if feature_config.tabular_features:
            num_tabular = len(feature_config.tabular_features)
            self.encoders["tabular"] = TabularEncoder(
                input_dim=num_tabular, hidden_dims=[256, 128], output_dim=128
            )
            self.embedding_dims.append(128)

        # Initialize categorical encoder
        if feature_config.categorical_features:
            # This should be configured based on actual vocabulary sizes
            vocab_sizes = {
                "gene_id": 30000,  # Approximate number of human genes
                "chromosome": 24,  # 22 + X + Y
                "strand": 2,  # + or -
            }
            self.encoders["categorical"] = CategoricalEncoder(
                vocab_sizes={
                    k: v
                    for k, v in vocab_sizes.items()
                    if k in feature_config.categorical_features
                },
                embedding_dim=32,
            )
            self.embedding_dims.append(
                32
                * len(
                    [f for f in feature_config.categorical_features if f in vocab_sizes]
                )
            )

        # Fusion layer
        self.fusion = AttentionFusion(input_dims=self.embedding_dims, output_dim=256)

        # Task-specific head
        if task == "binding_affinity":
            self.task_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),  # Regression output
            )
        elif task == "interaction_classification":
            self.task_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes or 2),
                nn.Softmax(dim=-1),
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(
        self,
        sequence_data: Optional[Dict[str, List[str]]] = None,
        tabular_data: Optional[torch.Tensor] = None,
        categorical_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model

        Args:
            sequence_data: Dictionary mapping sequence feature names to lists of sequences
            tabular_data: Tensor of tabular features [batch_size, num_features]
            categorical_data: Dictionary mapping categorical feature names to tensors of indices

        Returns:
            Task-specific output tensor
        """

        embeddings = []

        # Process sequence features
        if sequence_data:
            for feature_name, sequences in sequence_data.items():
                if feature_name in self.encoders:
                    emb = self.encoders[feature_name](sequences)
                    embeddings.append(emb)

        # Process tabular features
        if tabular_data is not None and "tabular" in self.encoders:
            emb = self.encoders["tabular"](tabular_data)
            embeddings.append(emb)

        # Process categorical features
        if categorical_data and "categorical" in self.encoders:
            emb = self.encoders["categorical"](categorical_data)
            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid input features provided")

        # Fuse embeddings
        fused = self.fusion(embeddings)

        # Apply task-specific head
        output = self.task_head(fused)

        return output


class FeatureValidator:
    """Validates feature-model compatibility"""

    COMPATIBILITY_MATRIX = {
        "sequence_transformer": {
            "valid": ["sequence", "tokenized_sequence", "text"],
            "invalid": ["continuous", "categorical_raw", "tabular"],
        },
        "cnn": {
            "valid": ["sequence", "image", "matrix", "grid"],
            "invalid": ["graph", "sparse_categorical", "raw_text"],
        },
        "tabular_nn": {
            "valid": ["continuous", "categorical_encoded", "normalized_numeric"],
            "invalid": ["raw_sequence", "raw_text", "image"],
        },
        "graph_nn": {
            "valid": ["graph", "adjacency_matrix", "edge_list"],
            "invalid": ["sequence", "tabular", "image"],
        },
    }

    @classmethod
    def validate(
        cls, features: Dict[str, str], model_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate feature compatibility with model type

        Returns:
            Tuple of (is_valid, list_of_issues)
        """

        if model_type not in cls.COMPATIBILITY_MATRIX:
            return False, [f"Unknown model type: {model_type}"]

        issues = []
        compatibility = cls.COMPATIBILITY_MATRIX[model_type]

        for feature_name, feature_type in features.items():
            if feature_type in compatibility["invalid"]:
                issues.append(
                    f"Feature '{feature_name}' of type '{feature_type}' "
                    f"is incompatible with {model_type}. "
                    f"Valid types: {compatibility['valid']}"
                )

        return len(issues) == 0, issues


class HybridPredictor:
    """
    High-level predictor that automatically selects the right model
    based on input features
    """

    def __init__(self):
        self.models = {}
        self.validator = FeatureValidator()

    def predict_binding_affinity(
        self,
        drug_smiles: str,
        protein_sequence: str,
        structural_features: Optional[Dict] = None,
    ) -> float:
        """
        Predict drug-protein binding affinity with proper feature handling
        """

        # Determine feature types
        features = {"drug_smiles": "sequence", "protein_sequence": "sequence"}

        sequence_data = {
            "drug_smiles": [drug_smiles],
            "protein_sequence": [protein_sequence],
        }

        tabular_data = None
        categorical_data = None

        if structural_features:
            # Separate structural features by type
            tabular_features = []
            categorical_features = {}

            for key, value in structural_features.items():
                if key in ["distance_to_stop_codon", "expression_level", "gc_content"]:
                    tabular_features.append(float(value))
                    features[key] = "continuous"
                elif key in ["gene_id", "chromosome", "strand"]:
                    # Convert to categorical indices (would need proper encoding)
                    categorical_features[key] = torch.tensor([0])  # Placeholder
                    features[key] = "categorical_encoded"

            if tabular_features:
                tabular_data = torch.tensor([tabular_features], dtype=torch.float32)

            if categorical_features:
                categorical_data = categorical_features

        # Validate feature compatibility
        if structural_features:
            # Use hybrid model for mixed features
            model_type = "hybrid"

            feature_config = FeatureConfig(
                sequence_features=["drug_smiles", "protein_sequence"],
                tabular_features=list(
                    filter(
                        lambda x: features.get(x) == "continuous",
                        structural_features.keys(),
                    )
                ),
                categorical_features=list(
                    filter(
                        lambda x: features.get(x) == "categorical_encoded",
                        structural_features.keys(),
                    )
                ),
            )

            model = HybridDrugDiscoveryModel(
                feature_config=feature_config, task="binding_affinity"
            )

            output = model(
                sequence_data=sequence_data,
                tabular_data=tabular_data,
                categorical_data=categorical_data,
            )

            return output.item()
        else:
            # Use sequence-only model for pure sequence data
            is_valid, issues = self.validator.validate(features, "sequence_transformer")

            if not is_valid:
                logger.warning(f"Validation issues: {issues}")

            # Fallback to existing DeepPurpose or similar model
            # This is where you'd integrate with existing models
            return self._predict_with_deeppurpose(drug_smiles, protein_sequence)

    def _predict_with_deeppurpose(
        self, drug_smiles: str, protein_sequence: str
    ) -> float:
        """Fallback to existing DeepPurpose model"""
        # Placeholder - integrate with actual DeepPurpose
        try:
            from DeepPurpose import utils, DTI as models

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
            return float(predictions[0])
        except Exception as e:
            logger.error(f"DeepPurpose prediction failed: {e}")
            return 0.0


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HybridPredictor()

    # Example 1: Sequence-only prediction
    affinity1 = predictor.predict_binding_affinity(
        drug_smiles="CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1",
        protein_sequence="MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW...",
    )
    print(f"Sequence-only prediction: {affinity1}")

    # Example 2: Prediction with structural features
    affinity2 = predictor.predict_binding_affinity(
        drug_smiles="CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1",
        protein_sequence="MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW...",
        structural_features={
            "gene_id": "ENSG00000135218",
            "chromosome": "12",
            "distance_to_stop_codon": 1500,
            "expression_level": 2.5,
            "gc_content": 0.45,
        },
    )
    print(f"Hybrid prediction: {affinity2}")

    # Example 3: Feature validation
    validator = FeatureValidator()

    # This should pass
    is_valid, issues = validator.validate(
        {"smiles": "sequence", "protein": "sequence"}, "sequence_transformer"
    )
    print(f"Sequence features with transformer: Valid={is_valid}")

    # This should fail
    is_valid, issues = validator.validate(
        {"smiles": "sequence", "gene_id": "categorical_raw"}, "sequence_transformer"
    )
    print(f"Mixed features with transformer: Valid={is_valid}, Issues={issues}")
