# Cross-domain knowledge transfer for bearing fault identification

This repository contains the code and resources for a machine learning project, where knowledge is transferred from labelled source domain to unlabelled target domain using Maximum Mean Discrepancy and pseudo labelling.

## Overview

The project implements a transfer learning approach to transfer knowledge about the bearing fault between different measurement domain (e.g., force measurements and sound pressure measurements), enabling accurate fault identification on the target domain, where labels are missing.

## Repository structure

```
.
├── measurements/
│   ├── measurements_description.txt
|   └── measurements.pkl
├── weights/
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

Raw measurements are located inside directory `measurements` in file `measurements.pkl`. For additional info see `measurements_description.txt` or associated research paper.

## License

Please refer to the associated research paper for citation and usage guidelines.