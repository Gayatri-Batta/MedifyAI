# 🩺 MedifyAI: Health Condition Prediction Chatbot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gayatri-Batta/MedifyAI/blob/main/MedifyAI.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/GenAI-LLM-purple)
![Fine-Tuning](https://img.shields.io/badge/Technique-QLoRA-orange)
![Optimization](https://img.shields.io/badge/Optimizer-Paged%20AdamW-green)

## Overview
**MedifyAI** is an intelligent Generative AI chatbot designed to predict potential health conditions based on user-described symptoms. 

This project addresses the challenge of running Large Language Models (LLMs) on consumer hardware. [cite_start]By leveraging **Parameter-Efficient Fine-Tuning (PEFT)** and **Quantization**, MedifyAI achieves high-performance domain adaptation for medical diagnostics while drastically reducing memory requirements[cite: 41, 42].

## Technical Deep Dive
This project implements advanced optimization techniques to enable fine-tuning on limited compute resources (e.g., a single GPU).

### 1. 4-Bit Quantization (QLoRA)
Standard LLMs (like Llama-2 or Mistral) require massive VRAM (approx. 14GB for a 7B model in FP16). 
* **Method:** We employed **NF4 (Normal Float 4)** quantization to load the model in 4-bit precision.
* [cite_start]**Impact:** This reduces the memory footprint by nearly **4x**, allowing the model to fit into ~4-6GB of VRAM while retaining near-original performance[cite: 42].

### 2. Low-Rank Adaptation (LoRA)
Instead of fine-tuning all 7 billion parameters (which is computationally expensive and prone to catastrophic forgetting), we used **LoRA**.
* **Method:** We freeze the pre-trained model weights and inject trainable rank decomposition matrices into the layers of the Transformer architecture.
* [cite_start]**Impact:** This reduces the number of trainable parameters by **~98%**, significantly speeding up training[cite: 42].

### 3. Paged AdamW Optimizer
Fine-tuning large models often leads to GPU Out-Of-Memory (OOM) errors during gradient spikes.
* **Method:** We utilized **Paged AdamW**, a custom optimizer from `bitsandbytes`.
* [cite_start]**Impact:** It manages memory peaks by offloading optimizer states to CPU RAM when the GPU memory limits are reached, ensuring stable training runs[cite: 42].

### 4. Loss Optimization
* **Method:** We optimized the model using **Cross-Entropy Loss**, focusing on minimizing the divergence between the predicted tokens and the ground-truth medical completions.
* [cite_start]**Impact:** Improved the model's ability to generate medically accurate and context-aware responses[cite: 43].

## 🛠️ Tech Stack
* **Core:** Python, PyTorch
* **Hugging Face:** `transformers`, `accelerate`, `peft` (Parameter-Efficient Fine-Tuning)
* **Optimization:** `bitsandbytes` (for quantization)
* **Data Processing:** Pandas, NumPy
* **Architecture:** Transformer-based LLM (Llama-2/Mistral architecture)

## Usage & Execution
This project is structured as a Jupyter Notebook for transparency and ease of experimentation.

### Prerequisites
* **Google Colab** (Recommended for free T4 GPU access) **OR**
* Local machine with NVIDIA GPU (minimum 8GB VRAM).

### Running the Notebook
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Gayatri-Batta/MedifyAI.git](https://github.com/Gayatri-Batta/MedifyAI.git)
    ```

2.  **Launch Jupyter**
    * Open `MedifyAI.ipynb` in Jupyter Lab or Google Colab.

3.  **Install Dependencies**
    * Run the initial cell to setup the environment:
        ```python
        !pip install torch transformers peft bitsandbytes accelerate
        ```

4.  **Execute Training Pipeline**
    * **Step 1:** Load Base Model (4-bit).
    * **Step 2:** Prepare Data (Tokenization).
    * **Step 3:** Initialize LoRA Configuration.
    * **Step 4:** Train/Fine-Tune.
    * **Step 5:** Inference (Chat with the bot).

## Results
* **Memory Efficiency:** Successfully fine-tuned a 7B parameter model on a single GPU (under 12GB VRAM).
* **Accuracy:** Enhanced symptom-to-condition mapping capabilities compared to the base zero-shot model.

## 📂 Project Files
* `MedifyAI.ipynb`: Main notebook containing the training pipeline and inference logic.
* `README.md`: Project documentation.
