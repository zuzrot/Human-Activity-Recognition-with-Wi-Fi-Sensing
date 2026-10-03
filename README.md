# CARE-BED: Wi-Fi CSI-Based Bedside Activity Monitoring

## Overview

CARE-BED (**CSI-based Activity Recognition for Elderly–Bedside Event Detection**) is a low-cost, privacy-preserving system for real-time bedside activity monitoring using Wi-Fi Channel State Information (CSI) acquired with ESP32 devices.

This repository accompanies the manuscript:

**“A Low-Cost Wireless Channel State Information System for Real-Time Nighttime Bedside Activity Monitoring of Older Adults”**

The system integrates:

- ESP32-based CSI acquisition,
- CSI frame validation and amplitude extraction,
- Hampel filtering and Savitzky–Golay smoothing,
- subcarrier selection,
- temporal segmentation,
- feature standardization,
- bidirectional long short-term memory (BiLSTM) classification,
- recording-disjoint offline evaluation,
- controlled ablation studies,
- and continuous real-time activity recognition.

CARE-BED focuses on four activity classes occurring within the monitored bedside area:

- `inactivity`
- `lying down`
- `sitting up`
- `fidgeting`

The current system is designed for single-person bedside monitoring. Walking, complete bed-exit events, falls, and room-scale mobility are outside the current classification scope.

---

## Repository Structure

```text
Human-Activity-Recognition-with-Wi-Fi-Sensing/
│
├── README.md
│
├── data_collection/
│   ├── check_csi_serial.py
│   └── collecting_data_sessions.py
│
├── datasets/
│   ├── ConfigurationA_dataset.zip
│   └── ConfigurationB_dataset.zip
│
├── notebooks/
│   ├── Models_analysis.ipynb
│   ├── CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb
│   ├── CARE_BED_A13_B12_filter_ablation_W20_recording_disjoint_5fold.ipynb
│   └── CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb
│
└── realtime/
    ├── CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb
    └── carebed_continuous_live_github.py
```

### File Description

#### `data_collection/`

- **`check_csi_serial.py`** – performs a short serial-stream test before data collection and displays the first received CSI lines.
- **`collecting_data_sessions.py`** – collects independent CSI recording sessions and stores raw CSI data, host-side timing information, and recording metadata.

#### `datasets/`

- **`ConfigurationA_dataset.zip`** – recordings collected under Configuration A.
- **`ConfigurationB_dataset.zip`** – recordings collected under Configuration B.

Together, the final dataset contains **25 independent recordings and 27,883 valid CSI frames**.

#### `notebooks/`

- **`Models_analysis.ipynb`** – exploratory comparison of deep-learning model families used during classifier selection.
- **`CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb`** – principal offline evaluation using 25-fold recording-disjoint leave-one-recording-out validation.
- **`CARE_BED_A13_B12_filter_ablation_W20_recording_disjoint_5fold.ipynb`** – fixed 5-fold recording-disjoint preprocessing-filter ablation.
- **`CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb`** – fixed 5-fold recording-disjoint temporal-window ablation.

#### `realtime/`

- **`CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb`** – trains and evaluates the single recording-disjoint deployment model subsequently used for real-time inference.
- **`carebed_continuous_live_github.py`** – continuous CARE-BED real-time inference pipeline using the frozen deployment model.

---

## Dataset

The final CARE-BED dataset contains:

- **25 independent continuous recordings**
- **27,883 valid CSI frames**
- **4 activity classes**
- **2 residential measurement configurations**
- recordings from a single adult participant

### Configuration A

Configuration A contains:

- **13 recordings**
- **12,026 valid CSI frames**
- 4 recordings of `inactivity`
- 3 recordings of `lying down`
- 3 recordings of `sitting up`
- 3 recordings of `fidgeting`

CSI acquisition was performed through Windows Subsystem for Linux 2 (WSL2) running Ubuntu, with the ESP32 USB interface forwarded using `usbipd-win`.

The serial connection operated at:

```text
115200 bps
```

The observed effective valid-CSI-frame rate was approximately:

```text
10.63 Hz
```

### Configuration B

Configuration B contains:

- **12 recordings**
- **15,857 valid CSI frames**
- 3 independent recordings per activity class

Acquisition was performed directly under Windows 11 using `pySerial`.

The serial connection operated at:

```text
921600 bps
```

The observed effective valid-CSI-frame rate was approximately:

```text
10.95 Hz
```

Configuration B was also used for the real-time evaluation.

---

## Hardware Setup

CARE-BED uses two ESP32 nodes forming a Wi-Fi sensing link:

- one ESP32 operates as the transmitter (TX),
- one ESP32 operates as the receiver (RX),
- the RX node is connected to the host computer through USB.

In both measurement configurations:

- TX–RX distance was approximately **3 m**,
- device height was approximately **0.35 m** above the floor,
- the devices operated with line-of-sight (LoS),
- the monitored bed area was positioned within the TX–RX sensing path.

The overall TX–RX and bed geometry was preserved between Configurations A and B, while the surrounding furniture arrangement differed.

---

## CSI Preprocessing

For each valid CSI frame, the processing pipeline:

1. validates the incoming `CSI_DATA` frame,
2. verifies the expected ESP32 MAC address,
3. uses the first 128 CSI I/Q values,
4. converts them into 64 CSI amplitude values,
5. applies Hampel filtering,
6. applies Savitzky–Golay smoothing,
7. removes predefined unreliable amplitude-feature positions,
8. retains 54 amplitude features,
9. forms fixed-length temporal segments,
10. standardizes the features using parameters fitted on training data only.

The removed zero-based feature indices are:

```python
[2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
```

The final deployed classifier input is therefore:

```text
20 frames × 54 features
```

### Filtering Parameters

Hampel filtering:

```text
window size = 10
```

Savitzky–Golay smoothing:

```text
window length = 10
polynomial order = 3
```

For the offline experiments, temporal windows are generated using a stride of:

```text
5 frames
```

---

## Final BiLSTM Classifier

The deployed CARE-BED classifier uses the following architecture:

```text
Input: 20 × 54
        │
        ▼
Bidirectional LSTM
64 units in each direction
128-dimensional output
        │
        ▼
Dropout
rate = 0.5
        │
        ▼
Dense Softmax
4 activity classes
```

### Training Configuration

- optimizer: `Adam`
- learning rate: `0.0005`
- loss: sparse categorical cross-entropy
- batch size: `8`
- early stopping patience: `5`
- best validation-loss weights restored

The model outputs probabilities for:

```text
inactivity
lying down
sitting up
fidgeting
```

---

## Evaluation Protocols

CARE-BED uses **three distinct recording-disjoint evaluation protocols**.

These protocols serve different purposes and should not be interpreted as interchangeable cross-validation results.

### 1. 25-Fold Leave-One-Recording-Out Evaluation

This is the **principal offline performance estimate**.

For each of the 25 folds:

- one complete recording is used as the test set,
- the remaining 24 recordings form the development set,
- within every development recording:
  - the first 80% of raw CSI frames are assigned to training,
  - the final 20% are assigned to validation,
- splitting is performed before preprocessing and segmentation,
- no segment crosses a recording or train–validation boundary,
- the test recording is excluded from training, validation, scaling, and model selection,
- a new BiLSTM model is trained independently in every fold.

This protocol is implemented in:

```text
notebooks/CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb
```

### 2. Fixed 5-Fold Recording-Disjoint Evaluation

The fixed 5-fold protocol is used **only for controlled ablation studies**.

The 25 recordings are divided into five fixed recording-disjoint folds. The same recording partitions are reused for every evaluated variant, allowing preprocessing and temporal-window configurations to be compared under identical data partitions.

This protocol is used in:

```text
notebooks/CARE_BED_A13_B12_filter_ablation_W20_recording_disjoint_5fold.ipynb
notebooks/CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb
```

### 3. Deployment Split

A separate recording-disjoint split is used to obtain **one fixed model for real-time deployment**.

The 25 recordings are divided into:

```text
20 recordings → development
5 recordings  → completely held-out test set
```

Within each of the 20 development recordings:

```text
first 80% of raw frames → training
final 20% of raw frames → validation
```

The five held-out recordings remain completely excluded from training, validation, feature-standardization fitting, and model selection.

The resulting model is evaluated on these five recordings and subsequently frozen for real-time use.

This procedure is implemented in:

```text
realtime/CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb
```

---

## Offline Results

### Principal 25-Fold Evaluation

The 25-fold recording-disjoint leave-one-recording-out evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | **77.51%** |
| Macro F1-score | **77.06%** |
| Weighted F1-score | **77.19%** |

Class-wise results:

| Activity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Inactivity | 0.892 | 0.948 | 0.919 |
| Lying down | 0.709 | 0.611 | 0.656 |
| Sitting up | 0.772 | 0.720 | 0.745 |
| Fidgeting | 0.715 | 0.815 | 0.762 |

---

## Ablation Studies

### Temporal-Window Ablation

Hampel filtering followed by Savitzky–Golay smoothing was kept fixed while the temporal-window length was varied.

| Window size | Accuracy | Macro F1-score |
|---|---:|---:|
| 10 frames | 74.90% | 74.47% |
| **20 frames** | **79.60%** | **79.09%** |
| 40 frames | 77.83% | 77.37% |

The 20-frame configuration achieved the highest accuracy and macro F1-score among the evaluated window lengths.

### Preprocessing-Filter Ablation

The temporal-window size was fixed to 20 frames.

| Preprocessing | Accuracy | Macro F1-score |
|---|---:|---:|
| No filtering | 78.44% | 77.73% |
| Hampel only | 79.18% | 78.63% |
| Savitzky–Golay only | 78.29% | 77.57% |
| **Hampel + Savitzky–Golay** | **79.60%** | **79.09%** |

The combined Hampel and Savitzky–Golay pipeline was retained in the final CARE-BED configuration.

---

## Deployment Model

The fixed deployment model is trained using the dedicated 20-development / 5-test recording split.

Performance on the five completely held-out recordings:

| Metric | Result |
|---|---:|
| Accuracy | **80.03%** |
| Macro precision | 80.09% |
| Macro recall | 78.31% |
| Macro F1-score | **77.35%** |
| Weighted F1-score | 79.49% |

Class-wise performance:

| Activity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Inactivity | 1.000 | 0.941 | 0.970 |
| Lying down | 0.624 | 0.425 | 0.506 |
| Sitting up | 0.623 | 1.000 | 0.768 |
| Fidgeting | 0.957 | 0.766 | 0.851 |

The same trained model, scaler, and label encoder are subsequently used unchanged for real-time inference.

---

## Real-Time Pipeline

The continuous real-time implementation is provided in:

```text
realtime/carebed_continuous_live_github.py
```

During operation, CARE-BED performs:

```text
ESP32 CSI stream
        │
        ▼
CSI frame validation
        │
        ▼
64 CSI amplitude values
        │
        ▼
Collect 20 valid frames
        │
        ▼
Hampel filtering
        │
        ▼
Savitzky–Golay smoothing
        │
        ▼
Remove unreliable features
64 → 54
        │
        ▼
Saved StandardScaler
        │
        ▼
BiLSTM inference
        │
        ▼
Activity probabilities
```

During continuous operation, each prediction uses one non-overlapping 20-frame segment. After a prediction is generated, the next 20 new valid CSI frames are collected for the next classifier input.

### Required Deployment Artifacts

Running the real-time script requires four artifacts generated by:

```text
realtime/CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb
```

The notebook generates:

```text
carebed_W20_HSG_single_split_live_model.keras
carebed_W20_HSG_single_split_scaler.pkl
carebed_W20_HSG_single_split_label_encoder.pkl
carebed_W20_HSG_single_split_metadata.json
```

Place these files in the same directory as the real-time script before running it.

---

## Real-Time Evaluation

The real-time experiment was conducted using Configuration B.

The evaluation contained:

- **80 randomized trials**
- **20 trials per activity**
- exactly one newly acquired 20-frame classifier input per trial
- no retraining or parameter adjustment based on live results

The frozen deployment model was used unchanged.

### Results

| Metric | Result |
|---|---:|
| Correct predictions | 59 / 80 |
| Accuracy | **73.75%** |
| Macro precision | 74.89% |
| Macro recall | 73.75% |
| Macro F1-score | **73.71%** |
| Weighted F1-score | 73.71% |

The dominant live error mode was confusion between `lying down` and `sitting up`.

Importantly, none of the 60 live trials corresponding to `lying down`, `sitting up`, or `fidgeting` was classified as `inactivity`.

---

## Runtime Characteristics

Software-level processing measurements reported in the study were:

| Stage | Mean time |
|---|---:|
| Parsing | 0.05 ms per CSI frame |
| Preprocessing | 83.15 ms per 20-frame segment |
| Inference | 67.48 ms per 20-frame segment |

For Configuration B, the effective accepted-frame rate was approximately:

```text
10.95 Hz
```

Therefore, acquiring the 20 valid CSI frames required for one classifier input takes approximately:

```text
1.83 s
```

CSI acquisition time and computational processing time are reported separately.

---

## Installation

Install the required Python libraries:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tensorflow joblib hampel pyserial jupyter
```

The real-time implementation was developed using Python 3.12.

---

## Data Collection

### 1. Check the CSI Serial Stream

Before recording data, the serial connection can be tested using:

```bash
python data_collection/check_csi_serial.py --port COM9 --baud 921600
```

For the WSL2 Configuration A setup, use the corresponding Linux serial device and baud rate, for example:

```bash
python data_collection/check_csi_serial.py --port /dev/ttyACM0 --baud 115200
```

### 2. Collect an Independent Recording

Example for Configuration B:

```bash
python data_collection/collecting_data_sessions.py \
    --activity sitting_up \
    --session 1 \
    --environment envB \
    --port COM9 \
    --baud 921600 \
    --mac YOUR_ESP32_MAC
```

The collector creates separate files for:

- raw CSI data,
- host-side timing information,
- recording metadata.

Independent recording sessions should use different session numbers.

---

## Running the Offline Experiments

Start Jupyter:

```bash
jupyter notebook
```

Then open the required notebook.

### Principal Offline Evaluation

```text
notebooks/CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb
```

### Filter Ablation

```text
notebooks/CARE_BED_A13_B12_filter_ablation_W20_recording_disjoint_5fold.ipynb
```

### Window-Length Ablation

```text
notebooks/CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb
```

### Model-Family Analysis

```text
notebooks/Models_analysis.ipynb
```

The `Models_analysis.ipynb` notebook documents the internal model-selection experiments. It should not be interpreted as the principal recording-disjoint performance estimate of the final CARE-BED classifier.

---

## Training the Deployment Model

Open:

```text
realtime/CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb
```

The notebook:

1. loads the Configuration A and Configuration B recordings,
2. constructs the fixed recording-disjoint development/test split,
3. performs the chronological training/validation split,
4. applies the final preprocessing pipeline,
5. trains the BiLSTM classifier,
6. evaluates it on the five held-out recordings,
7. saves the trained model and preprocessing artifacts required by the real-time script.

---

## Running Continuous Real-Time Recognition

After generating the deployment artifacts, run:

```bash
python realtime/carebed_continuous_live_github.py \
    --port COM9 \
    --mac YOUR_ESP32_MAC
```

The program continuously:

1. collects 20 new valid CSI frames,
2. performs the final preprocessing pipeline,
3. runs BiLSTM inference,
4. prints the predicted activity,
5. prints the class probabilities,
6. starts collecting the next non-overlapping segment.

Press `Ctrl+C` to stop the program.

---

## Reproducibility Note

The evaluation protocols in this repository have deliberately different purposes:

- **25-fold leave-one-recording-out**  
  provides the principal offline recording-level performance estimate;

- **fixed 5-fold recording-disjoint evaluation**  
  is used only for controlled preprocessing and temporal-window ablations;

- **single deployment split**  
  produces and evaluates the one frozen model subsequently used for real-time recognition.

The 25-fold and 5-fold results therefore correspond to different experimental protocols and should not be interpreted as two estimates obtained from the same cross-validation procedure.

---

## Related Resources

The implementation was informed by the following open-source Wi-Fi CSI sensing resources:

- **ESP32-WiFi-Sensing**  
  https://github.com/thu4n/ESP32-WiFi-Sensing

- **WiFi-CSI-Sensing-Benchmark**  
  https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark

---

## Citation

The formal citation for the accompanying article will be added after publication.

Until then, this repository can be referenced as:

```bibtex
@misc{carebed2026,
  author       = {Zuzanna Rotarska},
  title        = {CARE-BED: Wi-Fi CSI-Based Bedside Activity Monitoring},
  howpublished = {\url{https://github.com/zuzrot/Human-Activity-Recognition-with-Wi-Fi-Sensing}},
  year         = {2026}
}
```
