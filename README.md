# CARE-BED: Wi-Fi CSI-Based Bedside Activity Monitoring

## Overview

CARE-BED (**CSI-based Activity Recognition for Elderly–Bedside Event Detection**) is a low-cost, privacy-preserving system for real-time bedside activity recognition using Wi-Fi Channel State Information (CSI) acquired with ESP32 devices. The reported evaluation is a target-specific feasibility study with one adult participant in one residential room. Nighttime monitoring of older adults is the intended application; performance in older-adult populations, during uninterrupted overnight operation, or in unseen rooms has not been established.

This repository accompanies the manuscript:

**“A Low-Cost Wireless Channel State Information System for Real-Time Bedside Activity Monitoring”**

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

CARE-BED recognizes four predefined conditions within the monitored bedside area:

- `inactivity` — the monitored TX–RX sensing region is unoccupied;
- `lying down` — a maintained lying posture in bed, not the movement of lying down;
- `sitting up` — a maintained upright seated posture in bed, not the movement of sitting up;
- `fidgeting` — irregular low-amplitude body movements while lying in bed.

The current system is designed for single-person bedside monitoring. Walking, complete bed-exit events, falls, and room-scale mobility are outside the current classification scope. The experimental live protocol evaluated separately established conditions; it did not assess uninterrupted transitions between activities.

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
│   ├── CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb
│   ├── CARE_BED_ConfigurationA_timestamp_audit.py
│   └── CARE_BED_ConfigurationA_timing_by_unit.csv
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

- **`ConfigurationA_dataset.zip`** – archival Configuration A CSI data, including four activity-specific CSV files.
- **`ConfigurationB_dataset.zip`** – published Configuration B CSI dataset. The final evaluation uses three recording sessions (`s01`–`s03`) per activity.

Together, the final evaluation dataset contains **25 recording-level units and 27,883 valid CSI frames**: 13 units retrospectively reconstructed from archived Configuration A data and 12 independently recorded Configuration B sessions.

#### `notebooks/`

- **`Models_analysis.ipynb`** – exploratory comparison of deep-learning model families used during classifier selection.
- **`CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb`** – principal offline evaluation using 25-fold recording-disjoint leave-one-recording-out validation.
- **`CARE_BED_A13_B12_filter_ablation_W20_recording_disjoint_5fold.ipynb`** – fixed 5-fold recording-disjoint preprocessing-filter ablation.
- **`CARE_BED_A13_B12_window_ablation_W10_W20_W40_HSG_5fold.ipynb`** – fixed 5-fold recording-disjoint temporal-window ablation.
- **`CARE_BED_ConfigurationA_timestamp_audit.py`** – reproduces the archival Configuration A frame-rate and 20-frame elapsed-time calculations from ESP32 timestamps, including 32-bit timestamp wraparound.
- **`CARE_BED_ConfigurationA_timing_by_unit.csv`** – unit-level timing statistics produced by the timestamp audit.

#### `realtime/`

- **`CARE_BED_W20_HSG_single_recording_disjoint_live_model.ipynb`** – trains and evaluates the single recording-disjoint deployment model subsequently used for real-time inference.
- **`carebed_continuous_live_github.py`** – continuous CARE-BED real-time inference pipeline using the frozen deployment model.

---

## Dataset

The final CARE-BED dataset contains:

- **25 recording-level evaluation units** (13 reconstructed from archival Configuration A data; 12 independent Configuration B recordings)
- **27,883 valid CSI frames**
- **4 activity classes**
- **2 measurement configurations in one residential room**
- occupied activity conditions performed by **one adult participant** (not an older-adult cohort)

### Configuration A

Configuration A contains:

- **13 reconstructed recording-level units**
- **12,026 valid CSI frames**
- 4 recordings of `inactivity`
- 3 recordings of `lying down`
- 3 recordings of `sitting up`
- 3 recordings of `fidgeting`

Configuration A is an archival dataset from an earlier development phase. Its CSI data were preserved in four activity-specific merged CSV files. For the final evaluation, 13 recording-level units were reconstructed using discontinuities in the preserved ESP32 timestamps; explicit row-index boundaries are provided in the 25-fold LORO notebook. Preprocessing and window extraction were performed separately for each reconstructed unit, with no window crossing a unit boundary.

Acquisition used Windows Subsystem for Linux 2 (WSL2) running Ubuntu, with the ESP32 USB interface forwarded using `usbipd-win`.

The serial connection operated at:

```text
115200 bps
```

Retrospective analysis of local ESP32 timestamps yielded an effective rate of approximately **1.92 valid CSI frames/s**, computed from **12,026 frames** divided by the sum of first-to-last-frame elapsed intervals across the 13 reconstructed units. Frame arrivals were nonuniform, including short bursts separated by multi-second gaps.

For the generated 20-frame windows (stride 5), the **median elapsed time from the first to the twentieth frame was approximately 10.47 s**. These values describe the archived recordings; they are not a controlled measurement of sustained receiver throughput. The available data do not establish the technical cause of the observed gaps. The calculation is reproducible using `notebooks/CARE_BED_ConfigurationA_timestamp_audit.py`.

### Configuration B

Configuration B contains:

- **12 recordings**
- **15,857 valid CSI frames**
- 3 independent recordings per activity class (`s01`, `s02`, `s03`), selected from the original session archive

The acquisition order was rotated across these three series rather than recording all sessions of each class consecutively:

- **s01:** `inactivity` → `lying down` → `sitting up` → `fidgeting`
- **s02:** `lying down` → `sitting up` → `fidgeting` → `inactivity`
- **s03:** `sitting up` → `fidgeting` → `inactivity` → `lying down`

This reduces one potential acquisition-order confound but does **not** rule out session-specific CSI drift or other label-correlated conditions.

Acquisition was performed directly under Windows 11 using `pySerial`.

The serial connection operated at:

```text
921600 bps
```

The effective accepted-frame rate was approximately **10.95 Hz**, estimated from the number of accepted CSI frames and the host-measured recording duration. The ESP32 packet transmission rate was configured to 100 Hz. At this effective rate, collecting 20 valid frames corresponds to a **nominal buffer-filling time of approximately 1.83 s**; this is a rate-based estimate, not a directly measured first-to-last-frame interval.

**Only Configuration B was used for the 80-trial real-time evaluation.** The historical Configuration A conditions were not re-established for a separate live evaluation.

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

Windows are defined by **20 accepted frames**, not by a fixed elapsed duration. The pipeline does not perform timestamp-based resampling, compensate for irregular inter-frame intervals, or reject windows containing long timestamp gaps. Consequently, the physical duration of the input can differ between measurement configurations, and input-acquisition time should not be confused with computational inference time.

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

CARE-BED uses **three distinct evaluation protocols with recording-level separation between development and testing**. The 25 units comprise 13 reconstructed archival units from Configuration A and 12 separate recording sessions from Configuration B.

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

The five test units were selected by a **class-stratified random recording-level split** (`StratifiedShuffleSplit`, `test_size=5`, `random_state=42`), stratifying by activity rather than configuration. The test set contained **two inactivity units and one unit for each other class**, with **four units from Configuration B and one from Configuration A**. The 20 development units contained five units per activity class.

The five test units were excluded from deployment-model training, validation, feature-standardization fitting and weight selection. They had nevertheless been included in the separate five-fold ablation studies conducted on all 25 units, so the holdout was not independent of all earlier pipeline-design comparisons. The resulting single deployment model was evaluated on these five test units and subsequently frozen for real-time use.

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

The **unweighted mean accuracy across the 25 held-out units was 76.96% ± 27.03 percentage points (mean ± sample SD)**. Unlike the pooled accuracy of 77.51%, this unit-level mean gives equal weight to each held-out unit. From the same 25-fold held-out predictions, configuration-specific pooled accuracies were **77.47% (A)** and **77.54% (B)**; this does not isolate the effect of the very different frame-arrival patterns, because the acquisition setup and multipath conditions also differed.

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

| Window size | Pooled accuracy | Pooled macro F1 | Fold accuracy (mean ± SD) | Fold macro F1 (mean ± SD) |
|---|---:|---:|---:|---:|
| 10 frames | 74.90% | 74.47% | 75.07 ± 14.08% | 73.82 ± 14.89% |
| **20 frames** | **79.60%** | **79.09%** | 79.47 ± 12.20% | 77.61 ± 14.97% |
| 40 frames | 77.83% | 77.37% | 78.24 ± 12.05% | 76.80 ± 13.61% |

The 20-frame variant had the highest **pooled** scores, but fold-wise variability was substantial. Its paired accuracy difference versus 10 frames was **4.41 ± 3.57 percentage points** (better in four of five folds), and versus 40 frames **1.24 ± 4.19 percentage points** (better in two of five folds). Thus, 20 frames were retained as a practical compromise between temporal context and input-acquisition latency, **not as a statistically established superior choice**.

### Preprocessing-Filter Ablation

The temporal-window size was fixed to 20 frames.

| Preprocessing | Pooled accuracy | Pooled macro F1 | Fold accuracy (mean ± SD) | Fold macro F1 (mean ± SD) |
|---|---:|---:|---:|---:|
| No filtering | 78.44% | 77.73% | 78.61 ± 13.20% | 76.75 ± 15.23% |
| Hampel only | 79.18% | 78.63% | 79.15 ± 11.29% | 77.61 ± 12.98% |
| Savitzky–Golay only | 78.29% | 77.57% | 78.22 ± 12.17% | 76.08 ± 14.93% |
| **Hampel + Savitzky–Golay** | **79.60%** | **79.09%** | 79.47 ± 12.20% | 77.61 ± 14.97% |

The combined pipeline had the highest pooled scores, but the advantage over Hampel-only filtering was small relative to fold-wise variability: paired accuracy difference **0.33 ± 2.01 percentage points**, and paired macro F1 difference **0.01 ± 2.74 percentage points**. It performed better in only two of five folds. The combined pipeline was retained as a practical implementation choice, **not because a consistent or statistically significant superiority had been established**.

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

The pooled test accuracy of **80.03%** was calculated over **1,172 held-out test windows**. The **unweighted mean test accuracy across the five test units was 81.44% ± 23.44 percentage points (mean ± sample SD)**, illustrating substantial recording-to-recording variability. In this split, training comprised 3,440 windows and validation 811 windows from the 20 development units; the test set comprised 1,172 windows from the five held-out units.

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

- **80 live trials** organized into 20 repetitions, with all four conditions appearing once per repetition in randomized order
- **20 trials per activity**
- the participant first established the requested posture/movement condition; `inactivity` meant an unoccupied sensing region
- the serial input buffer was cleared after preparation and before collecting **20 new valid CSI frames** for one prediction
- no retraining or parameter adjustment based on live results
- no evaluation of uninterrupted transitions or trial-to-trial temporal smoothing

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

For **live accuracy**, the **95% Wilson confidence interval was 63.18–82.14%** (59/80 correct). For **macro F1**, the **95% percentile bootstrap confidence interval was 64.69–82.34%**, based on 100,000 class-stratified resamples (seed 42).

The dominant live error mode was confusion between `lying down` and `sitting up`: 10 of 20 `lying down` trials were predicted as `sitting up`, while 5 of 20 `sitting up` trials were predicted as `lying down`. Both `inactivity` and `fidgeting` were correctly recognized in 18 of 20 trials.

None of the 60 live trials corresponding to an occupied condition (`lying down`, `sitting up`, or `fidgeting`) was classified as `inactivity`. However, **live inactivity precision of 1.00 was based on 18 predictions**, with a **95% Wilson interval of 0.824–1.000**; inactivity recall was 0.90 (18/20; **95% Wilson interval 0.699–0.972**). The observed precision must not be interpreted as a guarantee of error-free operation.

---

## Runtime Characteristics

The following **stage-specific means were measured in Google Colab**, not on the Windows deployment host:

| Google Colab stage | Mean time |
|---|---:|
| Parsing | 0.05 ms per CSI frame |
| Preprocessing | 83.15 ms per 20-frame segment |
| Inference | 67.48 ms per 20-frame segment |

**Illustrative Windows 11 deployment-host measurements** shown in the manuscript console examples ranged from **approximately 185 to 267 ms for combined preprocessing and inference**. These were individual observations, not an average across all 80 live trials; the Colab stage breakdown and combined Windows-host measurements are not directly equivalent benchmarks.

**CSI input acquisition is separate from software processing.** In Configuration B, the effective rate of approximately 10.95 accepted frames/s gives a **nominal buffer-filling time of approximately 1.83 s** for 20 valid frames, excluding any subsequent processing. In historical Configuration A windows, the median observed first-to-last-frame interval was instead 10.47 s, reflecting nonuniform archival frame arrival. Equivalent real-time responsiveness under Configuration A was not evaluated.

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

Then open the required notebook. The published datasets are `datasets/ConfigurationA_dataset.zip` and `datasets/ConfigurationB_dataset.zip`. The four final experiment notebooks were originally written to load these data under the local source-archive filenames `envA_dataset.zip` and `carebed_new_data.zip`. **These are alternative input filenames, not additional datasets to upload to the repository.** Before reproducing an experiment, configure the notebook to use the published archives and verify that the expected internal directories (`envA_dataset/` and `carebed_new_data/`) are present. The final evaluation uses three Configuration B sessions (`s01`–`s03`) per activity.

To reproduce the archival timing verification (from the repository root):

```bash
python notebooks/CARE_BED_ConfigurationA_timestamp_audit.py --input datasets/ConfigurationA_dataset.zip --output notebooks/CARE_BED_ConfigurationA_timing_by_unit.csv
```

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

The `Models_analysis.ipynb` notebook documents **exploratory architecture selection** on a different, seven-class dataset with 200-frame windows. The preserved ViT experiment used a stratified random 80/20 segment-level split, with standardization fitted before the split; therefore these historical comparisons should **not** be treated as recording-disjoint benchmarks. GRU and BiLSTM showed comparable approximate accuracy in preliminary experiments, and BiLSTM was selected for suitable bidirectional window-based modelling, **not** for a demonstrated accuracy advantage over GRU. The principal final CARE-BED performance estimate is the separate 25-fold recording-disjoint experiment.

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

The 25-fold and 5-fold results therefore correspond to different experimental protocols and should not be interpreted as two estimates obtained from the same cross-validation procedure. Moreover, the **training and validation frame partitions come from different time ranges within the same development units**; only the outer test recordings are disjoint from the development set. The available experiments do **not** establish user-independent, cross-room, or uninterrupted overnight/transition recognition performance. The exploratory architecture-selection experiments in `Models_analysis.ipynb` are **not** the principal recording-disjoint evaluation of CARE-BED.

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
