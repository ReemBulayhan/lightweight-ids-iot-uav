# Lightweight Deep Learning for Multi-Attack Prediction in IoT and UAV Systems

## Project Structure

```
lightweight-ids-iot-uav/
│
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── cicids2017/
│   │   └── ciciot2023/
│   ├── processed/              # Cleaned & preprocessed data
│   └── merged/                 # Final merged dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_knowledge_distillation.ipynb
│   ├── 05_quantization.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading functions
│   │   └── preprocessing.py    # Preprocessing functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_eca.py          # Main model architecture
│   │   ├── teacher.py          # Teacher model for KD
│   │   └── student.py          # Student model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   └── distillation.py     # Knowledge distillation
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting functions
│
├── models/                     # Saved models
│   ├── checkpoints/
│   └── final/
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ReemBulayhan/lightweight-ids-iot-uav.git
cd lightweight-ids-iot-uav
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Datasets

### CICIDS2017
- **Source:** [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Size:** ~3 million records
- **Attacks:** DDoS, Brute Force, Botnet, Port Scan, etc.

### CIC-IoT2023
- **Source:** [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- **Size:** ~10 million records
- **Attacks:** Mirai, DoH tunneling, OS exploits, etc.

## Model Architecture

```
Input (Sequential Network Traffic Data)
    ↓
1D-CNN Layers
    ↓
ECA Block (Efficient Channel Attention)
    ↓
Global Average Pooling
    ↓
Dense Layers
    ↓
Output Layer (Multi-class Classification)
```

## Optimization Techniques

1. **Knowledge Distillation:** Teacher → Student model
2. **Quantization-Aware Training (QAT):** INT8 optimization

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Model Size (MB)
- Inference Time (ms)

## Author

**Reem Bulayhan Bandar Almutairi**  
Master's in Artificial Intelligence  
Imam Mohammad bin Saud Islamic University

## Supervisor

**Dr. Mostafa Ibrahim**

## License

This project is for academic purposes only.
