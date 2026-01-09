# INST0062 PIPELINE

A comprehensive toolkit for analyzing dense latents in Sparse Autoencoders (SAEs), implementing the "Dense Latents Are Features, Not Bugs" methodology on Gemma-2B with Matryoshka BatchTopK SAEs.

## Pipeline Components

### 1. Activation Collection (`activationcollector.py`)

**Purpose**: Collect SAE activations from transformer residual streams and compute density statistics.

**Usage**:
```bash
# Set environment variables
export LAYER=2
export TOKEN_BUDGET=1500000
export MAX_LENGTH=96
export BATCH_SIZE=32
export ACTS_JSON="dense_activations_layer_2.json"
export ACTS_NPZ="dense_activations_layer_2.npz"

# Run collection
python activationcollector.py
```

**Outputs**:
- `dense_activations_layer_X.npz`: Density statistics (densities, pos_counts, sum_vals, means_when_active)
- `dense_activations_layer_X.json`: Metadata and summary statistics

### 2. Threshold Justification (`antipodality/justify_threshold.py`)

**Purpose**: Generate statistical justification for the 0.05 density threshold used throughout the analysis.

**Key Features**:
- Analyzes density distributions across multiple layers (2, 12, 22)
- Generates histograms with threshold overlays
- Provides cross-layer consistency analysis
- Statistical validation of threshold effectiveness

**Usage**:
```bash
python antipodality/justify_threshold.py
```

**Outputs**:
- `threshold_layer_X.png`: Density histograms with threshold overlays showing 0.05 and 0.10 thresholds
- Console analysis with percentiles, feature counts, and cross-layer consistency metrics

### 3. Antipodality Analysis (`antipodality` module)

**Purpose**: Comprehensive geometric analysis of dense latent relationships in SAE weight space using a clean modular architecture.

**Architecture**:
- `antipodality/analysis.py`: Main computation
- `antipodality/viz/`: Visualization pipeline (payloads.py, plots.py, umap.py)
- `antipodality/io.py`: Data loading utilities
- `antipodality/pipeline.py`: Main orchestration
- `antipodality/cli.py`: Command-line interface

**Usage**:
```bash
# Run via module (recommended)
python -m antipodality dense_activations_layer_2.npz --layer 2

# Custom options
python -m antipodality dense_activations_layer_2.npz --layer 2 --density-threshold 0.05 --out-dir layer2_analysis
```

### Step 1: Collect Activations
```bash
# Process multiple layers sequentially
for LAYER in 2 12 22; do
    export LAYER=$LAYER
    export TOKEN_BUDGET=1500000
    export MAX_LENGTH=96
    export BATCH_SIZE=32
    export ACTS_JSON="dense_activations_layer_${LAYER}.json"
    export ACTS_NPZ="dense_activations_layer_${LAYER}.npz"
    python activationcollector.py
done
```

### Step 2: Validate Density Threshold
```bash
# Generate threshold justification plots and analysis
python antipodality/justify_threshold.py
```

### Step 3: Analyze Antipodal Relationships (New Modular Pipeline)
```bash
# Run comprehensive antipodality analysis for each layer using the new modular pipeline
python -m antipodality dense_activations_layer_2.npz --layer 2 --out-dir layer2_analysis
python -m antipodality dense_activations_layer_12.npz --layer 12 --out-dir layer12_analysis
python -m antipodality dense_activations_layer_22.npz --layer 22 --out-dir layer22_analysis

# Dials that can be changed
python -m antipodality dense_activations_layer_2.npz --layer 2 \
    --density-threshold 0.05 \
    --top-k-pairs 20 \
    --umap-neighbors 15 \
    --out-dir layer2_detailed
```

### Alternative: Quick Single Layer Analysis
```bash
# Minimal command for single layer (uses default output directory)
python -m antipodality dense_activations_layer_2.npz --layer 2
```

## Dependencies

### Core Requirements
```
torch==2.1.2
transformers>=4.46.0
sae-lens==6.5.1
numpy==1.26.4
matplotlib==3.10.5
seaborn==0.13.2
umap-learn>=0.5.9
datasets==3.6.0
scipy==1.16.1
scikit-learn==1.7.1
```

## Hardware Requirements

- **GPU**: A100/V100 for activation collection (model inference)

## Installation

```bash
# Clone the repository
cd <this repository>

# Install dependencies (recommended: use conda/mamba environment)
pip install -r requirements.txt

# Verify installation by running threshold analysis (uses only lightweight dependencies)
python antipodality/justify_threshold.py
```
