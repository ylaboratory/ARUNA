<img src="assets/aruna_logo.png" alt="ARUNA icon" width="150" height="150"/>


**ARUNA**: Slice-based self-supervised imputation enables upscaling of sequencing-based DNA methylation assays


## Overview

Whole-genome bisulfite sequencing (WGBS) provides near-comprehensive, base-resolution maps of DNA methylation, but its cost limits large-scale studies. Reduced representation bisulfite sequencing (RRBS) and related protocols offer cost-effective alternatives, but measure only a sparse subset of CpGs, creating substantial coverage mismatches across assays.

**ARUNA** is a self-supervised denoising convolutional autoencoder designed to *upscale sparse, sequencing-based methylomes to whole-genome resolution*. ARUNA operates on **methylation slices**—spatially stacked genomic windows that preserve local CpG correlation and cross-sample structure—allowing it to generalize across assays, donors, tissues, and datasets.

This repository provides:
1. Core ARUNA model and data-processing code.
2. Precomputed example datasets (GTEx chr21 subset).
3. Example notebooks for data preparation, inference, and training.
4. A pretrained model checkpoint for demonstration.



## Quick Navigation

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Data and Preprocessing](#data-and-preprocessing)
- [Running the Examples](#running-the-examples)
  - [Data creation](#data-creation)
  - [Inference with a Pretrained Model](#inference-with-a-pretrained-model)
  - [Training a Model](#training-a-model)
- [Notes on Reproducibility](#notes-on-reproducibility)
- [Citation](#citation)

---

## Installation


```bash
# Clone the repository:
git clone https://github.com/ylaboratory/ARUNA.git
cd ARUNA

# Create and activate a Python environment (Python ≥ 3.9 recommended):
conda create -n aruna python=3.11
conda activate aruna

# Install dependencies:
pip install -r requirements.txt
```

>This repository is not intended to be installed as a published Python package as of now.
All scripts and notebooks assume execution from the repository root.

## Repository Structure
```text
ARUNA/
├── aruna/                  # Core model and data-processing code
├── checkpoints/            # Pretrained model checkpoints
├── configs/                # Example configuration YAML files
├── data/
│   ├── gtex_subset/        # Example GTEx WGBS input data (chr21 only)
│   ├── metadata/           # Reference genome and RRBS metadata    
├── notebooks/              # Example notebooks 
├── scripts/                # Inference helpers
├── results/                # Output storage
└── requirements.txt
```
Large intermediate files and derived datasets are intentionally excluded from version control.

## Data and Preprocessing

### Bioinformatics pipeline

A reference WGBS preprocessing pipeline used to generate the input `.cov` files from FASTQ inputs is provided in: ```assets/bioinfopipe_pairedWGBS.sh```.


### Patch Creation
ARUNA operates on a patch-centric representation of the methylome and also creates chrom-centric data in the preprocessing pipeline.

#### Input data
Input data are BED-like CpG-level methylation files (e.g. Bismark *.cov) produced from the initial bioinformatics pipeline, organized as:

```text
data/gtex_subset/
└── <sample_id>/
    └── *.cpgMerged.CpG_report.merged_CpG_evidence.cov
```
An example GTEx subset (16 samples and chromosome 21 only) is included for demonstration.

#### Chrom-centric
- Shape: (num_cpgs, num_samples)
- Canonical CpG set derived from the reference genome (hg38)
- Stores fractional methylation and read depth

#### Patch-centric
- Fixed-size windows defined by number of CpGs per patch
- Enables convolutional modeling of local methylation structure


### Reference metadata
Located under: `data/metadata/`

Includes:
- Canonical CpG coordinates (hg38, 0-indexed)
- Chromosome lengths (hg38)
- Optional RRBS CpG observation probabilities for realistic rrbs-like missingness simulation


## Running the Examples

### Data creation
The subsequent step converts sample-wise methylation files into chrom-centric and patch-centric formats.

Relevant code:

- ```aruna/process_dataset.py```
- ```aruna/patch_metadata.py```

Example notebook: `notebooks/data_prep.ipynb`

Running this notebook produces:

```text
data/gtex/
├── chrom_centric/
│   ├── true/
│   ├── mcar_90/
│   └── rrbs_sim/
└── patch_centric/
    └── numCpg128/
        ├── true/
        ├── mcar_90/
        └── rrbs_sim/
```

### Inference with a Pretrained Model
- A pretrained ARUNA model is provided in: ```checkpoints/trained_model.pth```.
- You can run inference using: ```notebooks/example_infer.ipynb```.
- Predicted methylation values and evaluation artifacts will be written to: ```results/```.

### Training a Model

An example training workflow is provided in: `notebooks/example_train.ipynb`.

This notebook demonstrates:
- Loading and batching training data.
- Initializing and training ARUNA from scratch on simulated sparse methylomes.
- Saving model checkpoints for downstream inference to: ```checkpoints/```.

Note: Training is computationally intensive and was performed on GPU hardware for the experiments reported in the paper.

## Reproducibility Notes
- Example data are restricted to chromosome 21 for tractability.
- Noise simulation for inference is performed once and saved to disk.
- Exact Python dependencies are specified in requirements.txt.

## Citation
Coming Soon