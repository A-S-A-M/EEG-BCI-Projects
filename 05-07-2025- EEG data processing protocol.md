


### 🧠 **Seizure Detection Pipeline — Compact Plan**

#### **Data Summary**

- 📂 Total files: `5 channels × 100 samples = 500 text files`
    
- 📈 Each file: 4096 voltage readings
    
- 🎯 Goal: Detect seizures in EEG segments
    

---

### 🛠️ **Phase 1: Preprocessing**

1. **Read and structure data**
    
    - Read all `.txt` files
        
    - Organize into shape: `(samples, channels, timepoints)` = `(100, 5, 4096)`
        
2. **Bandpass Filtering**
    
    - Apply a **bandpass filter (0.5 – 40 Hz)** to remove drift and high-frequency noise
        
    - Use FIR/IIR filter (Butterworth preferred)
Code till here:
```
import os
import zipfile
import glob
import numpy as np
import scipy.signal as signal

# === Step 1: Design low-pass filter ===
fs = 173.16
cutoff = 40
b, a = signal.butter(4, cutoff / (fs / 2), btype='low')

# === Step 2: Define zip source and output folders ===
channels = ['F','O', 'N', 'S', 'Z']  # Remaining channels
base_zip_folder = 'EEGDATA'            # Contains O.zip, N.zip, etc.
unzipped_base = 'Unzipped_EEG'         # Where we'll extract .txt files
filtered_base = 'Filtered_EEG'         # Where filtered .txt will go

# === Step 3: Loop over each channel ===
for ch in channels:
    zip_path = os.path.join(base_zip_folder, f'{ch}.zip')
    extract_path = os.path.join(unzipped_base, ch)
    filtered_path = os.path.join(filtered_base, ch)

    # Ensure folders exist
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(filtered_path, exist_ok=True)

    # === Step 3a: Extract .zip ===
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print(f"✅ Extracted {zip_path} → {extract_path}")

    # === Step 3b: Filter each .txt file ===
    for file in sorted(glob.glob(os.path.join(extract_path, '*.txt'))):
        data = np.loadtxt(file)

        # Apply zero-phase Butterworth low-pass filter
        filtered_data = signal.filtfilt(b, a, data)

        # Save filtered result
        output_file = os.path.join(filtered_path, os.path.basename(file))
        np.savetxt(output_file, filtered_data)

        print(f"✅ Filtered: {file} → {output_file}")
```
        
3. **Artifact Removal** _(optional but useful)_
    
    - Detect flatlines, large spikes, or abnormal variance
        
    - Optionally apply Independent Component Analysis (ICA)

Artefact detection: We scan each filtered EEG signal for:

- **Flatlines:** very low variance (likely sensor dropout)
    
- **Spikes/Outliers:** sudden high-amplitude shifts
    
- **High-variance segments:** abnormally noisy signals
```
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Set threshold values
flatline_variance_thresh = 1e-6     # Below this is flatline
spike_threshold = 1000              # Above this (in µV) is a spike
high_variance_thresh = 500          # Above this (in µV²) is too noisy

# Input & output folders
input_root = 'Filtered_EEG'
output_cleaned = 'Cleaned_EEG'
os.makedirs(output_cleaned, exist_ok=True)

# Loop through channels
for ch in ['F','O','N','S','Z']:
    input_path = os.path.join(input_root, ch)
    output_path = os.path.join(output_cleaned, ch)
    os.makedirs(output_path, exist_ok=True)

    for file_path in sorted(glob.glob(os.path.join(input_path, '*.txt'))):
        signal_data = np.loadtxt(file_path)

        # === Artifact Detection ===
        var = np.var(signal_data)
        max_val = np.max(signal_data)
        min_val = np.min(signal_data)

        is_flatline = var < flatline_variance_thresh
        has_spike = (np.abs(signal_data) > spike_threshold).any()
        is_too_noisy = var > high_variance_thresh

        # If signal is clean, save it
        if not (is_flatline or has_spike or is_too_noisy):
            out_file = os.path.join(output_path, os.path.basename(file_path))
            np.savetxt(out_file, signal_data)
            print(f"✅ Clean: {file_path} → {out_file}")
        else:
            print(f"⚠️  Rejected: {file_path} | Flatline: {is_flatline}, Spike: {has_spike}, Noisy: {is_too_noisy}")

```
        
### 📊 Parameters & Their Effects

|Parameter|Effect When You **Tighten** It (lower threshold)|Effect When You **Loosen** It (higher threshold)|
|---|---|---|
|`FLATLINE_VAR_THRESHOLD`|More files marked as flatline (stricter)|Fewer flatlines detected|
|`SPIKE_THRESHOLD`|More files rejected for small spikes|Allows large amplitude jumps (may miss seizure spikes too)|
|`NOISE_VAR_THRESHOLD`|More files flagged as "noisy"|Accepts noisier signals (risk: garbage in → garbage out)|

---

### 🧠 Why Adjust Parameters?

- **Clinical-grade recordings**: Use tighter thresholds
    
- **Noisy, real-world data**: Use slightly relaxed thresholds
    
- **Research settings**: Tune based on experiment (e.g., seizure spike amplitude, typical noise levels)
---

### 📊 **Phase 2: Feature Extraction**

Extract features from each sample (per channel or across all):

1. **Time-Domain Features**
    
    - Mean, Std, Skewness, Kurtosis
        
    - Line length, zero-crossing rate
        
    - Hjorth Parameters
### ✅ 1. **Time-Domain Features**

For each 4096-point EEG signal (1D numpy array), compute:

|Feature|Description|
|---|---|
|Mean|Signal average|
|Std|Standard deviation|
|Skewness|Asymmetry of the distribution|
|Kurtosis|Tailedness of distribution|
|Line Length|Sum of absolute differences|
|Zero Crossings|Number of times signal crosses 0|
|Hjorth Params|Activity, Mobility, Complexity|
        
2. **Frequency-Domain Features**
    
    - Power Spectral Density (PSD)
        
    - Band power (Delta, Theta, Alpha, Beta, Gamma)
        
    - Spectral entropy
Time domain features:

```
import numpy as np
from scipy.stats import skew, kurtosis

def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return activity, mobility, complexity

def extract_time_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['line_length'] = np.sum(np.abs(np.diff(signal)))
    features['zero_crossings'] = ((signal[:-1] * signal[1:]) < 0).sum()
    activity, mobility, complexity = hjorth_parameters(signal)
    features['hjorth_activity'] = activity
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    return features
```
For implementing them in a single file:

```
sample = np.loadtxt('Cleaned_EEG/F/F006.txt')
time_feats = extract_time_features(sample)
print(time_feats)
```

### ✅ 2. **Frequency-Domain Features**

|Feature|Description|
|---|---|
|PSD|Power Spectral Density via Welch|
|Band Power|Power in Delta (0.5–4), Theta (4–8), Alpha (8–13), Beta (13–30), Gamma (30–40)|
|Spectral Entropy|Entropy of normalized PSD|
🔧 `extract_frequency_features(signal, fs=173.16)`

```
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def bandpower(frequencies, psd, fmin, fmax):
    idx_band = np.logical_and(frequencies >= fmin, frequencies <= fmax)
    return np.trapz(psd[idx_band], frequencies[idx_band])

def extract_frequency_features(signal, fs=173.16):
    features = {}
    freqs, psd = welch(signal, fs=fs, nperseg=512)

    total_power = np.trapz(psd, freqs)
    norm_psd = psd / total_power if total_power > 0 else psd

    features['delta_power'] = bandpower(freqs, psd, 0.5, 4)
    features['theta_power'] = bandpower(freqs, psd, 4, 8)
    features['alpha_power'] = bandpower(freqs, psd, 8, 13)
    features['beta_power']  = bandpower(freqs, psd, 13, 30)
    features['gamma_power'] = bandpower(freqs, psd, 30, 40)
    features['spectral_entropy'] = entropy(norm_psd)
    return features
```



        
3. **Nonlinear Features**
    
    - Approximate entropy
        
    - Fractal dimension
        
    - Hurst exponent
        
    - Energy entropy

🔧 `extract_nonlinear_features(signal)`


```
import numpy as np
import antropy as ant  # You may need: pip install antropy

def extract_nonlinear_features(signal):
    features = {}
    features['approx_entropy'] = ant.app_entropy(signal)
    features['fractal_dim'] = ant.petrosian_fd(signal)
    features['hurst_exp'] = ant.hurst_exp(signal)
    features['energy_entropy'] = ant.energy_entropy(signal)
    return features
```

Full Feature Extractor for One Signal

```
def extract_all_features(signal, fs=173.16):
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, fs))
    features.update(extract_nonlinear_features(signal))
    return features
```


### ✅ 3. **Nonlinear Features**

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| Approximate Entropy | Complexity/regularity         |
| Fractal Dimension   | Geometric irregularity        |
| Hurst Exponent      | Long-term memory              |
| Energy Entropy      | Signal energy across segments |
        

---
## 📁 Batch Processing: Extract Features for All Files

This will:

- Loop through all cleaned files
    
- Extract all features
    
- Save them to `features.csv` with `filename`, `channel`, and feature columns

```
import os
import glob
import pandas as pd

input_root = 'Cleaned_EEG'
channels = ['F', 'O', 'N', 'S', 'Z']
all_feature_rows = []

for ch in channels:
    ch_folder = os.path.join(input_root, ch)
    for file_path in sorted(glob.glob(os.path.join(ch_folder, '*.txt'))):
        signal = np.loadtxt(file_path)
        features = extract_all_features(signal)
        features['filename'] = os.path.basename(file_path)
        features['channel'] = ch
        all_feature_rows.append(features)

# Save to CSV
df = pd.DataFrame(all_feature_rows)
df.to_csv('eeg_features.csv', index=False)
print("✅ Features saved to eeg_features.csv")
```

```
import matplotlib.pyplot as plt

# Plot example signal from one channel (e.g., F)
file_path = 'Filtered_EEG/F/F001.txt'
signal = np.loadtxt(file_path)
time = np.arange(len(signal)) / 173.16  # Time axis in seconds

plt.figure(figsize=(12, 4))
plt.plot(time, signal, color='black')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (µV)')
plt.title('Frontal EEG Signal - Trial 001')
plt.grid(True)
plt.show()
```
![[Pasted image 20250705162609.png]]
### 🤖 **Phase 3: Seizure Detection**

- **Data Preparation**
    
    - Load the CSV
        
    - Split into training & test sets
        
    - Separate `X` (features) and `y` (labels)
        
- **Model Training**
    
    - Train a **Random Forest** as a solid baseline
        
    - (Optionally later: XGBoost, SVM, or deep learning)

        
- **Evaluation**
    
    - Accuracy, Precision, Recall, F1
        
    - Confusion Matrix
        
    - ROC-AUC Curve
        
- **Model Saving** (optional but useful for deployment)

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load Data ===
df = pd.read_csv('eeg_features_labeled.csv')

# Drop filename and channel (not features)
X = df.drop(columns=['filename', 'channel', 'label'])
y = df['label']

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 3. Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 4. Predict and Evaluate ===
y_pred = model.predict(X_test)
print("✅ Evaluation Metrics:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC  : {roc_auc_score(y_test, y_pred):.4f}")

# === 5. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Seizure", "Seizure"], yticklabels=["Non-Seizure", "Seizure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

### 📈 **Phase 4: Evaluation & Output**

### 1. **Confusion Matrix & ROC Curve**
```
✅ Evaluation Metrics:
Accuracy : 1.0000
Precision: 1.0000
Recall   : 1.0000
F1 Score : 1.0000
ROC AUC  : 1.0000
```
![[Pasted image 20250705163201.png]]
    
2. **Per-sample prediction**
    
    - Output: seizure probability per file
        
3. **Save Model**

```
import joblib

# Save the model to disk
joblib.dump(model, 'seizure_rf_model.pkl')
print("✅ Model saved as 'seizure_rf_model.pkl'")
```
    
    - Export trained model for future use
```
model = joblib.load('seizure_rf_model.pkl')
```
        

---

### 📁 Directory Structure (Suggested)

css

CopyEdit

`EEG_Seizure_Project/ ├── data/ │   ├── channel_1/ │   ├── channel_2/ │   ... ├── preprocessing/ ├── features/ ├── models/ ├── results/ └── main.py`

---


### 📊 **2. Visualize Feature Importance**
This shows you which features matter most — for example:

- Spectral entropy
    
- Alpha power
    
- Hjorth mobility
```
import matplotlib.pyplot as plt
import numpy as np

# Get feature names and importances
feature_names = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Determine how many features to plot (up to 20 or less)
top_n = min(20, len(feature_names))

# Plot top N features
plt.figure(figsize=(10, 6))
plt.title(f"Top {top_n} Feature Importances (Random Forest)")
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
plt.tight_layout()
plt.show()
```
![[Pasted image 20250705163859.png]]
### 🔧 Tools

- **Python:** `numpy`, `scipy`, `mne`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`
    
- **MATLAB:** `butter`, `filtfilt`, `pwelch`, `entropy`, `fitcsvm`, `fitcensemble`




## Advanced EEG - AI integration

## Seizure Detection Project — Execution Table

|**Phase**|**Title**|**Key Tasks**|**Output**|
|---|---|---|---|
|**1**|Deep Learning on Raw EEG|- Load raw `.txt` files  <br>- Normalize & reshape signals  <br>- Build 1D CNN, LSTM, or CNN-LSTM  <br>- Train + validate on labeled data|`deep_model.h5`  <br>ROC Curve  <br>Accuracy report|
|**2**|Classical Model Benchmarking|- Use `eeg_features_labeled.csv`  <br>- Train SVM, XGBoost, LightGBM  <br>- Compare with Random Forest  <br>- Plot performance bar chart|`benchmark_results.csv`  <br>Comparison plots|
|**3**|Autoencoder for Anomaly Detection|- Use only non-seizure data  <br>- Train deep autoencoder  <br>- Use reconstruction error as seizure signal  <br>- Evaluate with ROC curve|`autoencoder_model.h5`  <br>Error plots|
|**4**|Clinical Report Generator|- Extract top features & predictions  <br>- Generate structured `.txt` or `.pdf` reports per sample  <br>- Include confidence and feature highlights|`reports/<sample>_report.txt/.pdf`|
# Phase 1: Deep Learning on Raw EEG

**Step 1:** Load, normalize, reshape raw signals  
**Step 2:** Build and train 1D CNN  
**Step 3:** Evaluate and save model



## Step 1: Load, Normalize, and Reshape Raw EEG Data

We’ll now load filtered `.txt` EEG signals from all channels and prepare them for deep learning.

---

### ✅ Requirements:

- EEG folder: `Filtered_EEG/`
    
       
    `Filtered_EEG/ ├── F/ ├── O/ ├── N/ ├── S/ └── Z/`
    
- Each `.txt` file has 4096 voltage values
    
- `S`, `Z` folders = **seizure** → label `1`
    
- Others = **non-seizure** → label `0`

```
import os
import numpy as np
from sklearn.model_selection import train_test_split

# === Step 1: Base path and correct label logic ===
base_dir = 'Filtered_EEG'
seizure_folder = 'S'  # Only 'S' is labeled as seizure

X = []
y = []

# === Step 2: Loop through each channel folder ===
for ch in os.listdir(base_dir):
    ch_path = os.path.join(base_dir, ch)
    if not os.path.isdir(ch_path): continue

    label = 1 if ch == seizure_folder else 0

    for file in os.listdir(ch_path):
        if file.endswith('.txt'):
            path = os.path.join(ch_path, file)
            signal = np.loadtxt(path)

            if signal.shape[0] == 4096:  # Ensure correct length
                X.append(signal)
                y.append(label)

# === Step 3: Convert, normalize, reshape ===
X = np.array(X)
y = np.array(y)

# Normalize each signal: zero mean, unit variance
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# Reshape for CNN input (samples, timesteps, channels)
X = X[..., np.newaxis]

# === Step 4: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("✅ Phase 1A Complete: Raw EEG Loaded & Preprocessed")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"Positive samples: {np.sum(y)}, Total: {len(y)}")

```