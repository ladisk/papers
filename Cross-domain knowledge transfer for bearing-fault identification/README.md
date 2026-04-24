# Cross-domain knowledge transfer for bearing-fault identification

**The full implementation of the methods described in coresponding article, including all code and supporting materials, will be made publicly available upon acceptance of the article.**

This repository contains the code and resources for a machine learning project, where knowledge is transferred from labelled source domain to unlabelled target domain using Maximum Mean Discrepancy and pseudo labelling.

## Overview

The project implements a transfer learning approach to transfer knowledge about the bearing-fault between different measurement domains (e.g., contact force measurements and non-contact sound pressure measurements), enabling accurate fault identification on the target domain, where labels are missing.

## Repository structure

```
.
├── measurements/
│   ├── 0_measurements_description.txt  # Description of measurements
|   └── measurements_x.pkl              # Measurements from 1-272
├── Cross-domain_KT_pipeline            # Method pipeline
├── model_components.py                 # Model components
├── README.md                           # Info about project
└── requirements-txt                    # Python package dependencies
```

## How to use

### Prerequisites

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Training a Model

### Running Inference

### Using Pre-trained Models

## Measurements

Raw measurements are located inside directory `measurements` in file `measurements.pkl`. For additional info see `0_measurements_description.txt` or associated research paper.

## License

Please refer to the associated research paper for citation and usage guidelines.