# Parametric Visualization System for experiments

A flexible visualization system that allows configurable X and Y axes based on experimental variables.

## Usage

```bash
python create_visualizations.py --type <TYPE> [OPTIONS]
```

## Visualization Types

- `histogram`: Distribution histograms showing model predictions vs ground truth
- `heatmap`: Performance metrics displayed as heatmaps
- `individual`: Individual trial variability plots

## Available Variables (for X/Y axes)

- `verbal_intensity`: [none, mid, high]
- `facial_intensity`: [none, mid, high]  
- `source_specificity`: [not, region, very]
- `face_modality`: [img, text]

## Examples

### Basic histogram with default axes (verbal_intensity × facial_intensity)
```bash
python create_visualizations.py --type histogram
```

### Heatmaps for all metrics across source specificity and face modality (default)
```bash
python create_visualizations.py --type heatmap --x-axis face_modality --y-axis source_specificity
```

### Heatmap for specific metric only
```bash
python create_visualizations.py --type heatmap --x-axis face_modality --y-axis source_specificity --metric brier_score_mean
```

### Individual trial plots for facial intensity vs verbal intensity
```bash
python create_visualizations.py --type individual --x-axis verbal_intensity --y-axis facial_intensity
```

### Histogram comparing different experimental parameters
```bash
python create_visualizations.py --type histogram --x-axis source_specificity --y-axis face_modality --body-part wrist
```

### Flat layout with only X-axis (no Y-axis grouping)
```bash
python create_visualizations.py --type histogram --x-axis face_modality --y-axis none
```

## Parameters

- `--type`: Required. Visualization type (histogram, heatmap, individual)
- `--x-axis`: X-axis variable (default: facial_intensity)
- `--y-axis`: Y-axis variable (default: verbal_intensity, use 'none' for flat layout)
- `--metric`: Metric for heatmap (default: all - creates heatmaps for all metrics)
  - Available: brier_score_mean, entropy_model_mean, mae_mean, cosine_similarity_mean, accuracy, f1_score, all
- `--body-part`: Body part to analyze (default: wrist)
- `--results-file`: Path to results file (default: ../results/results.jsonl relative to script)
- `--output-dir`: Output directory (default: outputs)

## Output Structure

```
outputs/
├── histograms/           # Distribution comparison plots
├── heatmaps/            # Performance metric heatmaps  
└── individual_trials/   # Trial-by-trial variability plots
```

## Data Requirements

The system expects JSON Lines format with entries containing:
- `model_response`: JSON string with probability distribution
- `labels`: Ground truth labels with nested body part data
- `config_used`: Experimental configuration with variables

## Grid Layout

The visualization automatically adapts to the data:
- 3×3 grids for intensity variables (none/mid/high)
- 2×3 grids for modality×intensity combinations
- Other combinations scale accordingly