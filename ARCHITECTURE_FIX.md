# Architecture Fix: Addressing Feature-Model Mismatch in Drug Discovery Toolkit

## Problem Statement

There's a fundamental architectural inconsistency in mixing sequence-based transformer models with structural/tabular features. This document outlines the issue and provides solutions.

## Current Architecture Issues

### 1. Feature Type Mismatch
If using sequence-based models (like DNABERT-2, ChemBERTa, or protein language models), feeding them non-sequence features causes:
- **Information Loss**: Structural features like gene coordinates, distance to stop codons, and gene IDs cannot be properly processed by sequence transformers
- **Model Confusion**: Mixing feature types prevents the model from learning optimal representations
- **Suboptimal Performance**: The model either ignores structural features or incorrectly processes them

### 2. Current Implementation Analysis

The toolkit currently uses:
- **ChemBERTa**: Correctly processes SMILES sequences for molecular embeddings
- **DeepPurpose CNN**: Processes both drug SMILES and protein sequences
- **TamGen**: Generates molecules based on protein binding sites

These are generally appropriate, BUT if you're adding genomic features (gene IDs, stop codons, etc.), there's a mismatch.

## Recommended Solutions

### Solution 1: Hybrid Architecture (Recommended)

Create separate pathways for different data types:

```python
class HybridDrugDiscoveryModel:
    def __init__(self):
        # Sequence processors
        self.molecule_encoder = ChemBERTa()  # For SMILES
        self.protein_encoder = ProteinBERT()  # For amino acid sequences
        self.dna_encoder = DNABERT2()  # For genomic sequences
        
        # Tabular feature processor
        self.tabular_encoder = nn.Sequential(
            nn.Linear(n_tabular_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        # Fusion layer
        self.fusion = MultiModalFusion(
            sequence_dim=768,  # BERT output dimension
            tabular_dim=128,
            output_dim=512
        )
        
    def forward(self, smiles, protein_seq, genomic_seq=None, tabular_features=None):
        # Process each modality
        mol_emb = self.molecule_encoder(smiles)
        prot_emb = self.protein_encoder(protein_seq)
        
        embeddings = [mol_emb, prot_emb]
        
        if genomic_seq is not None:
            dna_emb = self.dna_encoder(genomic_seq)
            embeddings.append(dna_emb)
            
        if tabular_features is not None:
            tab_emb = self.tabular_encoder(tabular_features)
            embeddings.append(tab_emb)
            
        # Combine all embeddings
        combined = self.fusion(embeddings)
        return combined
```

### Solution 2: Feature Engineering for Sequence Models

If you must use sequence models with structural features, encode them into the sequence:

```python
def encode_structural_features_to_sequence(sequence, features):
    """
    Encode structural features as special tokens in the sequence
    """
    # Create special tokens for features
    feature_tokens = []
    
    if 'distance_to_stop' in features:
        distance_bin = discretize_distance(features['distance_to_stop'])
        feature_tokens.append(f"[DIST_{distance_bin}]")
    
    if 'gene_id' in features:
        # Use learned gene embeddings
        feature_tokens.append(f"[GENE_{features['gene_id']}]")
    
    if 'chromosomal_location' in features:
        chr_token = f"[CHR_{features['chromosomal_location']}]"
        feature_tokens.append(chr_token)
    
    # Prepend features to sequence
    augmented_sequence = ' '.join(feature_tokens) + ' [SEP] ' + sequence
    return augmented_sequence
```

### Solution 3: Proper Model Selection Based on Data Type

Choose models appropriate for your data:

| Data Type | Recommended Model | Feature Examples |
|-----------|-------------------|------------------|
| DNA/RNA Sequences | DNABERT-2, Nucleotide Transformer | ATCG sequences, splice sites |
| Protein Sequences | ESM-2, ProtBERT | Amino acid sequences |
| Molecular Structures | ChemBERTa, MolFormer | SMILES, InChI |
| Tabular Genomic Data | XGBoost, TabNet, MLP | Gene coordinates, expression levels |
| Mixed Data | Multi-modal architecture | All of the above |

## Implementation Guide

### Step 1: Identify Your Data Types

```python
def analyze_data_types(dataset):
    data_types = {
        'sequences': [],
        'tabular': [],
        'graphs': []
    }
    
    for feature_name, feature_data in dataset.items():
        if is_sequence(feature_data):
            data_types['sequences'].append(feature_name)
        elif is_tabular(feature_data):
            data_types['tabular'].append(feature_name)
        elif is_graph(feature_data):
            data_types['graphs'].append(feature_name)
    
    return data_types
```

### Step 2: Create Appropriate Encoders

```python
class MultiModalEncoder:
    def __init__(self, data_types):
        self.encoders = {}
        
        # Initialize appropriate encoder for each data type
        for seq_feature in data_types['sequences']:
            if 'dna' in seq_feature.lower():
                self.encoders[seq_feature] = DNABertEncoder()
            elif 'protein' in seq_feature.lower():
                self.encoders[seq_feature] = ProteinBertEncoder()
            elif 'smiles' in seq_feature.lower():
                self.encoders[seq_feature] = ChemBertaEncoder()
        
        # Tabular encoder for all tabular features
        if data_types['tabular']:
            self.encoders['tabular'] = TabularEncoder(
                n_features=len(data_types['tabular'])
            )
```

### Step 3: Implement Fusion Strategy

```python
class AttentionFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.attention_weights = nn.Linear(len(input_dims) * output_dim, len(input_dims))
        self.output_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, embeddings):
        # Project each embedding to common dimension
        projected = [layer(emb) for layer, emb in zip(self.attention_layers, embeddings)]
        
        # Calculate attention weights
        concat_proj = torch.cat(projected, dim=-1)
        weights = F.softmax(self.attention_weights(concat_proj), dim=-1)
        
        # Weighted sum
        weighted_sum = sum(w * p for w, p in zip(weights.unbind(-1), projected))
        
        return self.output_projection(weighted_sum)
```

## Specific Fixes for Current Codebase

### 1. Update `PredictBindingAffinity` Function

```python
def PredictBindingAffinity(
    input_data: Union[str, Dict],
    use_structural_features: bool = False
) -> float:
    """
    Enhanced binding affinity prediction with optional structural features
    """
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    
    # Sequence data
    x_drug = input_data.get("x_drug")  # SMILES
    x_target = input_data.get("x_target")  # Protein sequence
    
    # Optional structural features
    structural_features = None
    if use_structural_features:
        structural_features = {
            'gene_id': input_data.get('gene_id'),
            'chromosomal_location': input_data.get('chr_location'),
            'distance_to_stop_codon': input_data.get('stop_distance'),
            'expression_level': input_data.get('expression'),
        }
    
    if structural_features and any(structural_features.values()):
        # Use hybrid model
        model = HybridDTIModel()
        prediction = model.predict(
            drug_smiles=x_drug,
            protein_seq=x_target,
            structural_features=structural_features
        )
    else:
        # Use existing DeepPurpose model
        binding_model = models.model_pretrained(path_dir="DTI_model")
        X_pred = utils.data_process(
            [x_drug], [x_target], [7.635],
            drug_encoding="CNN",
            target_encoding="CNN",
            split_method="no_split"
        )
        predictions = binding_model.predict(X_pred)
        prediction = float(predictions[0])
    
    return prediction
```

### 2. Add Feature Validator

```python
def validate_feature_model_compatibility(features, model_type):
    """
    Ensure features are compatible with the model architecture
    """
    compatibility_matrix = {
        'sequence_transformer': {
            'valid': ['sequence', 'tokenized_sequence'],
            'invalid': ['tabular', 'continuous', 'categorical_raw']
        },
        'cnn': {
            'valid': ['sequence', 'image', 'matrix'],
            'invalid': ['graph', 'sparse_categorical']
        },
        'tabular_nn': {
            'valid': ['tabular', 'continuous', 'categorical_encoded'],
            'invalid': ['raw_sequence', 'raw_text']
        }
    }
    
    issues = []
    for feature_name, feature_type in features.items():
        if feature_type in compatibility_matrix[model_type]['invalid']:
            issues.append(
                f"Feature '{feature_name}' of type '{feature_type}' "
                f"is incompatible with {model_type}"
            )
    
    if issues:
        raise ValueError("\n".join(issues))
    
    return True
```

## Testing the Fix

```python
def test_architecture_consistency():
    """
    Test that the architecture properly handles different feature types
    """
    # Test 1: Sequence-only features
    result1 = PredictBindingAffinity({
        'x_drug': 'CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1',
        'x_target': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW...'
    })
    assert result1 is not None
    
    # Test 2: Mixed features (should use hybrid model)
    result2 = PredictBindingAffinity({
        'x_drug': 'CC(C)C1=NC(=CS1)CN(C)C(=O)NC1=CC=CC=C1',
        'x_target': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRW...',
        'gene_id': 'ENSG00000135218',
        'chr_location': '12q24.31',
        'stop_distance': 1500,
        'expression': 2.5
    }, use_structural_features=True)
    assert result2 is not None
    
    # Test 3: Incompatible features should raise error
    try:
        validate_feature_model_compatibility(
            {'gene_id': 'categorical_raw', 'sequence': 'sequence'},
            'sequence_transformer'
        )
        assert False, "Should have raised error"
    except ValueError:
        pass  # Expected
    
    print("All architecture consistency tests passed!")
```

## Migration Plan

1. **Phase 1**: Add validation to existing functions
   - Add `validate_feature_model_compatibility()` checks
   - Log warnings when mixed features are detected

2. **Phase 2**: Implement hybrid architecture
   - Create `HybridDTIModel` class
   - Add tabular encoder for structural features
   - Implement attention-based fusion

3. **Phase 3**: Update API and documentation
   - Update function signatures to clarify feature types
   - Add examples of proper feature usage
   - Create migration guide for existing users

## Performance Monitoring

```python
class ArchitectureMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_prediction(self, features_used, model_type, prediction_time, accuracy=None):
        self.metrics['feature_types'].append(self._classify_features(features_used))
        self.metrics['model_type'].append(model_type)
        self.metrics['prediction_time'].append(prediction_time)
        if accuracy:
            self.metrics['accuracy'].append(accuracy)
    
    def analyze_performance(self):
        """Identify which feature-model combinations work best"""
        df = pd.DataFrame(self.metrics)
        return df.groupby(['feature_types', 'model_type']).agg({
            'prediction_time': 'mean',
            'accuracy': 'mean'
        }).sort_values('accuracy', ascending=False)
```

## Conclusion

The key to fixing the architecture mismatch is to:
1. **Separate** different types of features into appropriate processing pipelines
2. **Use** the right model for each feature type
3. **Combine** representations intelligently using fusion techniques
4. **Validate** that features and models are compatible

This approach ensures that each type of data is processed optimally while still allowing for multi-modal learning when needed.