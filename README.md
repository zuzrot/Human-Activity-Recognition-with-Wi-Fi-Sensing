# CARE-BED: CSI-Based Elderly Activity Recognition

## Overview

CARE-BED is a low-cost, privacy-preserving Wi-Fi sensing system for real-time human activity recognition (HAR) in indoor environments. This repository contains the implementation developed as part of the engineering thesis **"Human Activity Recognition with Wi-Fi Sensing"** at AGH University of Science and Technology in Krakow.

The system is designed for elderly activity monitoring, especially for bedside and nighttime scenarios where potentially risky events such as getting out of bed, sitting up, or unusual movement patterns may be important. Instead of relying on cameras or wearable devices, CARE-BED uses **Wi-Fi Channel State Information (CSI)** collected with **ESP32** devices.

The repository contains an end-to-end pipeline for:
- CSI acquisition,
- preprocessing and segmentation,
- deep learning model evaluation,
- training of the final classifier,
- real-time activity inference.

**Why it exists**

The goal of the project is to explore whether a low-cost ESP32-based Wi-Fi sensing system can support real-time, privacy-preserving activity recognition in residential environments, with a particular focus on applications relevant to elderly care.

**Link to paper**

The paper based on this project is not publicly available yet.

---

## Features

- CSI acquisition on ESP32
- Real-time preprocessing using Hampel filtering and Savitzky–Golay smoothing
- Deep learning inference using BiLSTM
- Five-class HAR for elderly care scenarios
- Real-time serial streaming and CSI frame parsing
- Subcarrier selection and feature standardization
- Deployment-oriented validation in a residential indoor environment

---

## Repository Structure

```text
CARE-BED/
├── README.md
├── collecting_data.py
├── bilstm_5ac.py
├── 5_activities_bilstm_model.ipynb
├── Models_analysis.ipynb
├── dataset.zip
└── Human_Activity_Recognition_with_Wi_Fi_Sensing_notes
### File description
* **`collecting_data.py`** – CSI acquisition from the serial stream and saving measurements to CSV
* **`bilstm_5ac.py`** – real-time preprocessing and inference pipeline using the trained BiLSTM model
* **`5_activities_bilstm_model.ipynb`** – notebook for training and evaluation of the final five-activity BiLSTM model
* **`Models_analysis.ipynb`** – notebook for comparative analysis of different deep learning architectures
* **`dataset.zip`** – dataset used in experiments
* **`Human_Activity_Recognition_with_Wi_Fi_Sensing_notes`** – project notes and supporting documentation
```
---

## Installation
It is recommended to use a dedicated Python environment.

### Conda
```bash
conda create -n care-bed python=3.12
conda activate care-bed
pip install numpy pandas scipy scikit-learn pyserial matplotlib tensorflow notebook jupyter
```
### Virtual environment
```bash
python -m venv care-bed
source care-bed/bin/activate   # Linux / macOS
# Windows: care-bed\Scripts\activate
pip install numpy pandas scipy scikit-learn pyserial matplotlib tensorflow notebook jupyter
```
The project was developed and tested using Python 3.12.

## Running Offline Experiments

The offline experiments are currently notebook-based.

Start Jupyter:

```bash
jupyter notebook
```

Then open:

- `Models_analysis.ipynb` – for model comparison
- `5_activities_bilstm_model.ipynb` – for the final five-class BiLSTM model

The final classifier is a **BiLSTM** model with:

- input shape: `20 × 54`
- one bidirectional LSTM layer with `64` units
- dropout `0.5`
- softmax output for `5` classes

### Training setup

- optimizer: `Adam`
- learning rate: `0.0005`
- loss: `sparse categorical cross-entropy`
- early stopping with patience `5`

> **Note:** there is currently no standalone `train.py` script in the repository. Training is implemented in the notebooks.

## Running Real-Time Pipeline

The real-time pipeline is implemented in:

```bash
python bilstm_5ac.py
```

During runtime, the system performs the following steps:

1. loads the trained BiLSTM model and scaler,
2. opens the serial connection to the RX ESP32 node,
3. validates incoming CSI frames,
4. parses complex CSI values and converts them to amplitude,
5. buffers the data into windows of `20` frames,
6. applies Hampel and Savitzky–Golay filtering,
7. standardizes the segment,
8. performs inference and outputs the predicted activity label.

> **Note:** there is currently no standalone `run_realtime.py` script in the repository. The real-time functionality is implemented in `bilstm_5ac.py`.

## Dataset Format

The final model operates on CSI amplitude segments of shape **20 × 54**.

### Input format

Each input sample consists of:

- `20` consecutive CSI frames,
- `54` selected amplitude-based features per frame.

The original ESP32 CSI output contains values for `64` subcarriers, but unreliable edge subcarriers are removed during preprocessing, resulting in `54` retained features.

### Preprocessing steps

- raw CSI acquisition from ESP32,
- conversion from complex CSI to amplitude,
- Hampel filtering,
- Savitzky–Golay smoothing,
- subcarrier reduction from `64` to `54`,
- temporal segmentation into windows of `20` frames,
- feature standardization using a fitted scaler.

### Activity classes

The system recognizes five activity classes:

- `walking`
- `lying down`
- `sitting up`
- `fidgeting`
- `no activity`

### Example

```python
segment.shape == (20, 54)
```

## Hardware Setup

The system was validated in a residential indoor environment using two ESP32-based nodes:

- one acting as **transmitter (TX)**,
- one acting as **receiver (RX)**.

The receiver was connected to a laptop via USB and used as the CSI data sink.

### Experimental setup

- **TX–RX distance:** approximately `3 m`
- **Device height:** approximately `0.35 m` above the floor
- **Configuration:** line-of-sight (LoS)
- **Activity zone:** around the midpoint between TX and RX
- **Scenario:** activities performed on a single-person mattress placed directly on the floor

### Host environment

The runtime environment used in the experiments:

- AMD Ryzen 7 6800H
- 16 GB RAM
- Windows 11
- WSL2 with Ubuntu 24.04.3 LTS
- USB forwarding with `usbipd-win`
- serial device exposed as `/dev/ttyACM*`
- serial acquisition via `pySerial` at `115200 bps`

## Results Summary

The system was evaluated both offline and in real-time residential conditions.

### Offline evaluation

- custom dataset: **2,983 CSI segments**
- train/test split: **80/20**
- final BiLSTM accuracy: **0.96**

### Real-time evaluation

- real-time effectiveness: approximately **80%**
- prediction update interval: approximately **10 s**

### Notes

The performance gap between offline and real-time conditions is mainly related to:

- short transitional movements,
- environmental perturbations affecting multipath propagation,
- similarity between subtle recumbent activities and static states.

## Related Resources

This project was developed with inspiration from and partial methodological support from the following repositories:

- **ESP32-WiFi-Sensing**  
  `https://github.com/thu4n/ESP32-WiFi-Sensing/tree/master`

- **WiFi-CSI-Sensing-Benchmark**  
  `https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark`

These resources were helpful for CSI acquisition, preprocessing concepts, and baseline experimentation.

## Thesis

This repository accompanies the engineering thesis:

**Zuzanna Rotarska, _Human Activity Recognition with Wi-Fi Sensing_, AGH University of Science and Technology in Krakow, 2025.**

## Citation

If you use this repository, please cite the engineering thesis:

```bibtex
@misc{rotarska2025thesis,
  author       = {Zuzanna Rotarska},
  title        = {Human Activity Recognition with Wi-Fi Sensing},
  year         = {2025},
  note         = {Engineering Thesis, AGH University of Science and Technology in Krakow}
}
```

You may also cite the repository itself:

```bibtex
@misc{har_wifi_repo,
  author       = {Zuzanna Rotarska},
  title        = {Human Activity Recognition with Wi-Fi Sensing},
  howpublished = {\url{https://github.com/zuzrot/Human-Activity-Recognition-with-Wi-Fi-Sensing}},
  year         = {2025},
  note         = {Accessed: 2025-01-15}
}
```

### Related references

```bibtex
@article{natarajan2023machine,
  title={{A machine learning approach to passive human motion detection using WiFi measurements from commodity IoT devices}},
  author={Natarajan, Anisha and Krishnasamy, Vijayakumar and Singh, Munesh},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={72},
  pages={1--10},
  year={2023},
  publisher={IEEE}
}

@article{yang2023sensefi,
  title={{SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing}},
  author={Yang, Jianfei and Chen, Xinyan and Zou, Han and Lu, Chris Xiaoxuan and Wang, Dazhuo and Sun, Sumei and Xie, Lihua},
  journal={Patterns},
  volume={4},
  number={3},
  year={2023},
  publisher={Elsevier}
}

@article{ma2019wifi,
  title={{WiFi sensing with channel state information: A survey}},
  author={Ma, Yongsen and Zhou, Gang and Wang, Shuangquan},
  journal={ACM Computing Surveys (CSUR)},
  volume={52},
  number={3},
  pages={1--36},
  year={2019},
  publisher={ACM New York, NY, USA}
}

@article{hernandez2022wifi,
  title={{Wifi sensing on the edge: Signal processing techniques and challenges for real-world systems}},
  author={Hernandez, Steven M and Bulut, Eyuphan},
  journal={IEEE Communications Surveys \& Tutorials},
  volume={25},
  number={1},
  pages={46--76},
  year={2022},
  publisher={IEEE}
}

@misc{esp32_wifi_sensing_repo,
  author       = {Thuận Tống et al.},
  title        = {{ESP32-WiFi-Sensing}},
  year         = {2020},
  howpublished = {\url{https://github.com/thu4n/ESP32-WiFi-Sensing/tree/master}},
  note         = {Accessed: 2025-01-10}
}

@misc{wifi_csi_sensing_benchmark_repo,
  author       = {Xinyan Chen, Jianfei Yang},
  title        = {{WiFi-CSI-Sensing-Benchmark}},
  year         = {2021},
  howpublished = {\url{https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark}},
  note         = {Accessed: 2025-01-10}
}
```
