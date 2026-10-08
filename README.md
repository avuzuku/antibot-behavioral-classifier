# Behavioral Anti-Bot Detection System (PyTorch + evdev)

A low-level mouse dynamics behavioral biometrics system designed to distinguish human users from automated scripts in real time under modern Linux environments (Wayland and X11).

> **Academic Project Note:**  
> This repository contains a university coursework project (*Курсовой проект*) developed for academic evaluation.  
> **Localization Notice:** The console interface, interactive terminal prompts, and session reports are localized in **Russian**.

---

## Overview
Traditional active challenge-response tests (such as CAPTCHAs) introduce friction into the user experience and are increasingly bypassed by modern computer vision models. This project explores passive, continuous behavioral biometrics as an alternative.

By capturing raw hardware interrupts directly from the Linux kernel input subsystem (`evdev`), the system analyzes microscopic trajectory dynamics (speed variations, acceleration noise, and path curvature) and classifies movement in real time using a lightweight PyTorch neural network.

---

## Key Features

- **Wayland Compatibility:** Operates completely independently of display servers and desktop environments by reading character devices directly from `/dev/input/event*`.
- **Atomic Event Synchronization (`EV_SYN`):** Fixes the "staircase" trajectory distortion and high-polling-rate speed spikes by grouping relative axis deltas (`REL_X`, `REL_Y`) on kernel sync barriers.
- **Kernel-Level Bot Emulation (`uinput`):** Features a dedicated hardware bot simulator creating a virtual device (`Virtual Bot Mouse`) via the Linux `uinput` driver for authentic black-box integration testing.
- **Engineered Feature Space:** Extracts 5 mathematical metrics per 50-point sliding window:
  - Mean velocity (`mean_speed`)
  - Velocity standard deviation / motor noise (`std_speed`)
  - Mean acceleration (`mean_accel`)
  - Acceleration standard deviation (`std_accel`)
  - Trajectory straightness coefficient (*Straightness*)
- **Balanced Deep Learning Pipeline:** Employs Random Undersampling to prevent majority-class bias, scales features via `StandardScaler`, and trains a 3-layer Multi-Layer Perceptron (MLP).
- **Real-Time Moving-Average Smoothing:** Integrates a sliding-history consensus filter (15-window buffer) to eliminate instantaneous false positives and provides an end-of-session summary report upon exit (`Ctrl+C`).

---

## Project Structure

| File | Description |
|---|---|
| `collector_evdev.py` | Low-level event logger capturing synchronized hardware coordinates to CSV. |
| `uinput_bot.py` | Virtual kernel-level bot simulator generating algorithmic linear motions. |
| `preprocess.py` | Sliding window feature extractor converting raw coordinates to a mathematical dataset. |
| `train.py` | Class balancer, scaler, and PyTorch MLP training script (`antibot_model.pth`, `scaler.pkl`). |
| `detector_evdev.py` | Real-time interactive classifier featuring moving-average smoothing and session auditing. |

---

## Technical Stack & Architecture

- **Language:** Python 3
- **Deep Learning:** PyTorch (`torch.nn`, `torch.optim`)
- **System Interface:** `python-evdev` (`/dev/input/` and `/dev/uinput`)
- **Data Science:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Neural Network Architecture:**
  - Input Layer: 5 features
  - Hidden Layer 1: 16 units + ReLU
  - Hidden Layer 2: 8 units + ReLU
  - Output Layer: 1 unit + Sigmoid (Binary Cross-Entropy Loss, Adam Optimizer)

---

## Installation & Setup

### 1. Requirements
Ensure your Linux user belongs to the `input` group to access input device files:
```bash
sudo usermod -aG input $USER
newgrp input
```

### 2. Dependencies
Install the required packages in your Python virtual environment:
```bash
pip install torch evdev pandas numpy scikit-learn joblib
```

---

## Usage Workflow

### 1. Collect Telemetry
Record 1–2 minutes of natural human movement:
```bash
sudo python collector_evdev.py
```
*(Stop recording with `Ctrl+C` to save `mouse_human.csv`).*

Switch the collector target and run the kernel bot simulator in parallel:
```bash
sed -i 's/mouse_human.csv/mouse_bot.csv/g' collector_evdev.py
sudo python collector_evdev.py
```
*(In a separate terminal):*
```bash
sudo python uinput_bot.py
```

### 2. Feature Extraction & Training
Preprocess the collected logs into a 5-feature dataset and train the model:
```bash
python preprocess.py
python train.py
```

### 3. Real-Time Detection
Run the real-time classifier:
```bash
sudo python detector_evdev.py
```
The console will display the running statistics (`Str` for straightness, `StdS` for velocity variance) along with an active status verdict (`ЧЕЛОВЕК` or `ОБНАРУЖЕН БОТ!`). Pressing `Ctrl+C` prints the cumulative session verdict.
