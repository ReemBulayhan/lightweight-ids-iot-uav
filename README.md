# SkyShield-IDS: Lightweight Multi-Class Intrusion Detection Across Enterprise and IoT Traffic

Code, trained models and results of the master's thesis *Development of Lightweight Deep Learning Architectures for Multi-Attack Prediction in IoT and Drone Systems* (Imam Mohammad Ibn Saud Islamic University) and of the related journal manuscript.

The study examines whether a single lightweight model can detect multiple attack types in traffic from two different network environments. CICIDS2017 (enterprise) and CIC-IoT2023 (IoT), which share no feature names, are aligned through thirteen semantically unified features and an eight-class taxonomy (BENIGN, Botnet, BruteForce, DDoS, DoS, PortScan, Spoofing, WebAttack). Duplicate feature vectors are removed before splitting, so that no test vector occurs in the training data.

## Models

| Model | Description | Parameters (merged) |
|---|---|---|
| SkyShield-IDS | 1D-CNN (64 and 128 filters) with Efficient Channel Attention after each convolution, global average pooling and a skip branch (Model E of the ablation study) | 44,110 |
| SkyShield-IDS-S | Student with every width divided by 4, trained with knowledge distillation (T = 3, alpha = 0.4) | 3,358 |
| SkyShield-IDS-Q8, SkyShield-IDS-S-Q8 | Full-integer INT8 TensorFlow Lite versions (post-training quantization) | |

The model definitions are in [`src/models.py`](src/models.py).

## Main results (merged configuration, test split)

| Result | Value |
|---|---|
| SkyShield-IDS, Macro F1 | 0.598 (0.609 ± 0.010 over three seeds) |
| SkyShield-IDS, Macro F1 with post-hoc decision offsets | 0.681 (false alarm rate 22.7% to 12.3%) |
| SkyShield-IDS-S, Macro F1 | 0.583 (0.584 ± 0.001 over three seeds); 9.0 KB RAM and about 4 ms per record estimated on an Arduino Nicla Vision (Cortex-M7) |
| Merged model against single-dataset models on the same 13 features | within 0.024 to 0.043 Macro F1 |
| Cross-dataset transfer (13 features, 7 classes) | Macro F1 drops by 93% to 97% |
| Dataset-of-origin test (XGBoost on the 13 features) | 100.0% accuracy |
| Compact XGBoost 20×3 (about 3,500 stored values), tuned offsets | Macro F1 0.675, the same as SkyShield-IDS-S |

The full results are reported in the thesis and the manuscript; the JSON files in [`results/`](results/) contain the per-run values.

## Repository structure

```
notebooks/   Colab notebooks of the experimental phases (see below)
src/         Model definitions (Models A to E, student, distillation loss, class weights)
models/
  keras/     Trained Keras models: final_* (SkyShield-IDS), cross_* (cross-dataset), kd_* (SkyShield-IDS-S), nokd_* (student without distillation)
  tflite/    TensorFlow Lite float32 and INT8 models of the teachers and distilled students
results/     Per-run results of the ablation, final training, distillation, quantization and feature-shift analysis
```

The raw CSV files are expected under `MyDrive/Thesis/data/cicids2017/` and `MyDrive/Thesis/data/ciciot2023/`.

Configurations: `config1_merged` (13 unified features, 8 classes), `config2_cicids2017` (62 features, 7 classes) and `config3_ciciot2023` (37 features, 8 classes).

## Notebooks

| Notebook | Content |
|---|---|
| 01_data_cleaning_and_splits | Loading and cleaning of the raw CSV files (missing and infinite values, exact duplicates), mapping of the labels to the eight unified classes and initial stratified splits |
| 02a_feature_engineering_and_unification | Feature sets of the three configurations, including the thirteen semantically unified features of the merged configuration |
| 02b_duplicate_vectors_and_final_splits | Removal of invalid values and duplicate feature vectors, majority labels for conflicting vectors, final 64/16/20 stratified splits and refitted scaling |
| 03_model_variants | Models A to E and a timing test |
| 04_ablation_study | Three-fold cross-validation of the five variants in the three configurations and model selection |
| 05_final_training_and_cross_dataset | Final training of SkyShield-IDS and the cross-dataset evaluation |
| 06_feature_distribution_shift | Kolmogorov–Smirnov analysis of the unified features |
| 07_knowledge_distillation | SkyShield-IDS-S with and without distillation |
| 08_int8_quantization | Full-integer INT8 post-training quantization and CPU latency |
| 09_qat_feasibility | Compatibility check of quantization-aware training (tfmot) |
| 10_tree_baselines_and_origin_test | XGBoost and Random Forest baselines, compact XGBoost models, decision offsets for the baselines and the dataset-of-origin test |
| 11_repeated_runs_and_training_variants | Repeated runs (seeds 1 and 2) in the merged configuration, early stopping on Macro F1 and softened class weights |
| 12_decision_offsets_and_int8_calibration | Post-hoc decision offsets and INT8 calibration variants |
| 13_repeated_runs_all_configurations | Repeated runs for the remaining experiments (prepared; not yet run) |

The microcontroller estimates were obtained with Edge Impulse (EON Compiler) for the Arduino Nicla Vision from the TensorFlow Lite files in `models/tflite/`; they are platform estimates, not measurements on a physical board.

## Data

The datasets are not included. They are available from the Canadian Institute for Cybersecurity:

* CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html (CSV release, 78 features)
* CIC-IoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html (merged CSV release with 39 features)

## Running the notebooks

The notebooks were run on Google Colab (NVIDIA L4 GPU for training, CPU runtime for quantization and latency) and read and write Google Drive paths of the form `/content/drive/MyDrive/Thesis/outputs_v3/`. To run them elsewhere, change the path constants in the first code cell of each notebook. Training notebooks save a checkpoint after every epoch and resume after an interruption.

```bash
git clone https://github.com/ReemBulayhan/lightweight-ids-iot-uav.git
cd lightweight-ids-iot-uav
pip install -r requirements.txt
```

## Author

Reem Bulayhan Bandar Almutairi, M.Sc. in Artificial Intelligence, Imam Mohammad Ibn Saud Islamic University

Supervisor: Dr. Mostafa Ibrahim

## License

MIT License (see [LICENSE](LICENSE)).
