# A Hybrid Approach for Detecting AI-Generated Text in English

**Based on COLING 2025 GenAI Content Detection Task 1 (Subtask A - English)**

---

## Overview

This project addresses the challenge of distinguishing between AI-generated and human-written English text. We propose a **hybrid detection model** that combines the strengths of train-based discriminative classifiers (DeBERTa) and train-free statistical methods (DNA-DetectLLM) to improve detection robustness and performance across diverse domains and generation models.

Our hybrid approach consistently outperforms individual baseline methods on both validation and test datasets, demonstrating the effectiveness of integrating complementary detection signals.

---

## Key Features

- **Hybrid Architecture**: Combines DeBERTa-v3-base with DNA-DetectLLM repair scores
- **Improved Performance**: Leverages both learned representations and statistical features
- **Lightweight LLMs**: Uses GPT-2 and Pythia models for efficient perplexity computation
- **Comprehensive Evaluation**: Performance analysis across domains, sources, and generation models

---

## Repository Structure
- `notebook_llm_A.ipynb` # Experiments using GPT-2 variants for repair score computation, output results can be found in [colab](https://colab.research.google.com/drive/1Wg68g8i1EYDjFdypclZwDnmGBa9lWGhm?usp=sharing)
- `notebook_llm_B.ipynb` # Experiments using Pythia variants for repair score computation, output results can be found in [colab](https://colab.research.google.com/drive/1GLI5tp-NCae6W3FXzJjHnSH7hc91cZ3I?usp=sharing)
- ReadMe.md

### Notebooks

Both notebooks contain complete experimental pipelines including:
- **Data preprocessing** with source-stratified proportional sampling
- **Baseline implementations** (DNA-DetectLLM and DeBERTa-based classifier)
- **Hybrid model training** and evaluation
- **Detailed results** with visualizations and performance breakdowns
- **Analysis** by sub-source/domain, and generation model
---
## Dataset

We use subsets of the **COLING 2025 MGT Detection Dataset (English Subtask)** with balanced sampling to preserve the original distribution while managing computational constraints.

**Access the dataset:**
- [Download from Google Drive](https://drive.google.com/drive/folders/1_Ohnu5P3MKi_jBCebDPi0Z0_4Mc9r3WF?dmr=1&ec=wgc-drive-hero-goto)
- Includes preprocessed data, sampled datasets, and intermediate results
- Dataset paths and structure are documented in the notebooks

---

## Results Summary

Our hybrid model achieves significant improvements over baseline methods:

| Model | Dev F1 | Dev Accuracy | Test F1 | Test Accuracy |
|-------|--------|--------------|---------|---------------|
| DNA-DetectLLM (GPT) | 74.61% | 60.34% | 73.81% | 63.06% |
| DNA-DetectLLM (Pythia) | 74.52% | 60.20% | 70.14% | 55.72% |
| DeBERTa-v3-base | 84.07% | 81.26% | 73.93% | 67.46% |
| **Hybrid-GPT** | **96.50%** | **95.60%** | **80.10%** | **76.52%** |
| **Hybrid-Pythia** | **96.43%** | **95.52%** | **80.42%** | **77.02%** |

Detailed analysis including performance by source, domain, and generation model is available in the notebooks and project report.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Google Colab (recommended for GPU access)

### Usage

1. **Download the dataset** from [Google Drive](https://drive.google.com/drive/folders/1_Ohnu5P3MKi_jBCebDPi0Z0_4Mc9r3WF?dmr=1&ec=wgc-drive-hero-goto)

2. **Open the notebooks** in Google Colab or Jupyter:
   - `notebook_llm_A.ipynb` for GPT-2-based experiments
   - `notebook_llm_B.ipynb` for Pythia-based experiments

3. **Follow the instructions** within each notebook to:
   - Configure data paths
   - Run baseline models
   - Train and evaluate the hybrid model
   - Analyze results

All dependencies and detailed execution steps are documented within the notebooks.

---

## Methodology

Our hybrid model integrates:

1. **Train-Free Component**: DNA-DetectLLM repair score using lightweight LLMs
2. **Train-Based Component**: DeBERTa-v3-base with modified classification head
3. **Feature Fusion**: Concatenation of DeBERTa pooled output (768-dim) and repair score (1-dim)
4. **Fine-Tuning**: Efficient training with frozen backbone and trainable classification head

For detailed methodology, refer to the project report.

---

## Acknowledgements

This project builds upon the following works:

- **[DNA-DetectLLM](https://github.com/Xiaoweizhu57/DNA-DetectLLM)**: Train-free detection method using perplexity-based repair scores
- **[Advacheck AI Detector](https://github.com/Advacheck-OU/ai-detector-coling2025)**: DeBERTa-based classifier architecture (COLING 2025 Task 1 winner)
- **COLING 2025 GenAI Content Detection Task**: Dataset and task framework

---

## License

This project is for **academic use only**.

---
