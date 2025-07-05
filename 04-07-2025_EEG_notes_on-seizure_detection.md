## 🔁 Recap — The Path You’re Walking:

|Tier|Name|Goal|
|---|---|---|
|1️⃣|Raw Signal Basics|Understand EEG waveform|
|2️⃣|Signal Features|Quantify activity, noise, power|
|3️⃣|Baseline|Define calm brain state|
|4️⃣|Frequency Analysis|Measure brainwave band activity|
|5️⃣|Seizure Detection|Detect abnormalities using Z-score or features|
## **Imports and Setup**

python

CopyEdit

`import numpy as np import scipy.signal as signal import glob import os`

- `numpy` (as `np`): Used for numerical operations, including loading and saving arrays.
    
- `scipy.signal`: Provides signal processing tools like filters.
    
- `glob`: Finds all file paths matching a pattern (used to find all `.txt` files).
    
- `os`: Used for handling folders and file paths.

## ⚙️ **Set Parameters**


`fs = 173.16  # Sampling rate in Hz cutoff = 40  # Low-pass filter cutoff in Hz`

- `fs`: Sampling rate of the EEG data. You said each signal was sampled at 173.16 Hz.
    
- `cutoff`: We want to **retain only the signals below 40 Hz** (this includes Delta, Theta, Alpha, and low Beta bands).
    
    - This removes muscle noise, powerline interference, and high-frequency artifacts.
        

---

## 🧠 **Design Butterworth Low-pass Filter**

b, a = signal.butter(4, cutoff / (fs / 2), btype='low', analog=False)`

- `signal.butter(...)`: Designs a **Butterworth filter**.
    
    - **Order 4**: Defines sharpness of the filter roll-off. Higher orders = sharper but less stable.
        
    - `cutoff / (fs/2)`: Normalize the cutoff frequency. Nyquist frequency is `fs/2`.
        
    - `btype='low'`: Low-pass filter.
        
    - `analog=False`: Ensures it's a digital filter.
        

It returns:

- `b`: Numerator coefficients of the filter.
    
- `a`: Denominator coefficients.
    

---

## 📁 **Prepare Input and Output Folders**

`input_folder = 'path_to_txt_files'  # Replace with your folder path output_folder = 'filtered_txt_files' os.makedirs(output_folder, exist_ok=True)`

- `input_folder`: Folder where your raw `.txt` EEG files are stored.
    
- `output_folder`: Destination to save filtered files.
    
- `os.makedirs(..., exist_ok=True)`: Creates the output folder if it doesn’t already exist.
    

---

## 🔁 **Loop Over Each File and Process**

python

CopyEdit

`for file in glob.glob(os.path.join(input_folder, '*.txt')):`

- `glob.glob(...)`: Grabs **all `.txt` files** in the folder.
    
- `os.path.join(...)`: Makes sure file paths work on any OS (Windows, Linux, Mac).
    

---

## 📥 **Load EEG Data**

python

CopyEdit

`data = np.loadtxt(file)`

- `np.loadtxt`: Loads EEG signal from `.txt` file as a NumPy array.
    
- Assumes the file is a **single-column text file** (i.e., a 1D time series).
    

---

## 🧽 **Apply Filter**

python

CopyEdit

`filtered_data = signal.filtfilt(b, a, data)`

- `filtfilt`: Applies the filter **forward and backward** to avoid phase distortion.
    
- This preserves the **shape and timing of brain waveforms**, which is **crucial in EEG**.
    

---

## 💾 **Save the Filtered Signal**

python

CopyEdit

`output_path = os.path.join(output_folder, os.path.basename(file)) np.savetxt(output_path, filtered_data)`

- `os.path.basename(file)`: Extracts just the file name (not full path).
    
- `np.savetxt`: Writes the filtered data back to a `.txt` file.
    

---

## ✅ **Optional Confirmation**

python

CopyEdit

`print(f"Filtered {file} → saved to {output_path}")`

- Just a message to show progress — useful when working with hundreds of files like in your case (4096 files).
    

---

## ✅ Summary

|Step|Purpose|
|---|---|
|Set sampling and filter parameters|Define your filter characteristics|
|Design Butterworth filter|Create a stable and smooth low-pass filter|
|Loop through EEG files|Batch processing of all EEG data|
|Load → Filter → Save|Clean your EEG signals and store them for further analysis|
## 🧠 What Is a Seizure in EEG?

A **seizure** is a burst of abnormal electrical activity in the brain. On EEG, this typically appears as:

|Feature|What to Look For|
|---|---|
|**Sudden onset**|Signal changes abruptly from normal to spiky/synchronous patterns|
|**High amplitude spikes**|Large, sharp voltage spikes or waves (often >100 µV)|
|**Rhythmic bursts**|Repetitive, periodic waves at a fixed frequency (like 3 Hz or 6 Hz)|
|**Evolution over time**|The pattern may change in frequency or amplitude as the seizure develops|
|**Stereotypy**|Looks similar across multiple events (if there are repeated seizures)|
|**Multiple channels**|Generalized seizures affect multiple brain regions/channels|
## 🧪 Step-by-Step Guide to Analyzing Seizures in Your Data

### 🔹 Step 1: **Visual Inspection**

Start with plotting filtered EEG signals from different channels:

python

CopyEdit

`import matplotlib.pyplot as plt  # Plot example signal from one channel (e.g., F) file_path = 'Filtered_EEG/F/001.txt' signal = np.loadtxt(file_path) time = np.arange(len(signal)) / 173.16  # Time axis in seconds  plt.figure(figsize=(12, 4)) plt.plot(time, signal, color='black') plt.xlabel('Time (s)') plt.ylabel('Voltage (µV)') plt.title('Frontal EEG Signal - Trial 001') plt.grid(True) plt.show()`

### What to look for visually:

- Sudden appearance of **repetitive spikes** or **sharp waves**
    
- Periods of **sustained rhythmicity**
    
- High amplitude (>2-3x baseline)
    
- Contrast with normal background noise (which is usually irregular)
    

---

### 🔹 Step 2: **Statistical and Spectral Features**

You can compute the following features per trial or time window:

| Feature                           | What it indicates                                    |
| --------------------------------- | ---------------------------------------------------- |
| **Energy** (`np.sum(signal**2)`)  | High energy → likely seizure burst                   |
| **Variance or STD**               | High variance = high amplitude spikes                |
| **Line length**                   | Total amount of signal activity (good for seizures)  |
| **Spectral Power (e.g., 3–5 Hz)** | Spike-and-wave discharges appear around 3 Hz         |
| **Entropy**                       | Seizure may reduce signal randomness (lower entropy) |
Example:

python

CopyEdit

`def extract_features(signal, fs):     energy = np.sum(signal ** 2)     variance = np.var(signal)     line_length = np.sum(np.abs(np.diff(signal)))     return energy, variance, line_length`

---

### 🔹 Step 3: **Frequency Analysis (e.g., Power Spectral Density)**

python

CopyEdit

`from scipy.signal import welch  f, psd = welch(signal, fs=173.16, nperseg=1024)  plt.semilogy(f, psd) plt.title('Power Spectral Density') plt.xlabel('Frequency (Hz)') plt.ylabel('Power') plt.grid(True) plt.show()`

- Seizures often have **narrowband peaks** (3–6 Hz for absence seizures, 8–12 Hz for rhythmic discharges).
    
- Compare the PSD during a suspected event vs baseline.
    

---

### 🔹 Step 4: **Sliding Window Approach**

You can divide the signal into windows (e.g., 2 sec) and compute features in each:

python

CopyEdit

`window_size = int(2 * 173.16)  # 2 seconds features = []  for i in range(0, len(signal) - window_size, window_size):     segment = signal[i:i + window_size]     feats = extract_features(segment, 173.16)     features.append(feats)`

This lets you find **when** a seizure might be happening.

---

### 🔹 Step 5: **Multiple Channels**

You should eventually compare features across:

- `F`, `N`, `O`, `S`, `Z`
    

Seizures can be **focal** (in one region) or **generalized** (across many). If multiple channels show the same abnormal pattern → likely generalized.

---

## 🚫 What Is NOT a Seizure?

- Sudden movement artifacts (short, spiky, irregular)
    
- Eye blinks (frontal, slow upward curve, <1 sec)
    
- Muscle activity (high-frequency noise)
    
- Flatline periods (could be power disconnection, not seizure)
    

---

## 🛡️ Final Advice

|DO ✅|DON'T ❌|
|---|---|
|Plot and explore the signal visually|Blindly merge all trials|
|Compare abnormal vs normal|Trust only amplitude|
|Use features + time windowing|Filter without saving intermediate output|
|Start with 1–2 trials to learn|Assume all high-amplitude = seizure|

---

## 🔮 Want to Go Further?

Once you visually and numerically understand a few seizure vs non-seizure trials, we can:

- Train an **SVM or CNN** to classify them
    
- Do **time-frequency analysis** (Short-Time Fourier or Wavelet)
    
- Build a **real-time detection model** using NumPy/PyTorch
    

---

When you're ready, show me a sample signal (or trial you suspect has seizure), and I’ll help you analyze it directly. We’ll look at it together — frame by frame.


## 🧠 What is Baseline in EEG?

The **baseline** is a reference signal or time window that represents "normal" or "resting" brain activity. It’s what you compare other signals against — to detect changes (like seizures, responses to stimuli, or cognitive load).

---

## 🎯 Why You Need It

|Use Case|Baseline Role|
|---|---|
|Seizure detection|Compare suspicious segment to non-seizure period|
|ERP (event-related potentials)|Compare post-stimulus to pre-stimulus|
|Power spectral analysis|Normalize band powers|
|Feature extraction|Subtract or divide by baseline variance, energy, etc.|

---

## ✅ How to Define Baseline (for Your Case)

You have multiple `.txt` files per channel. Here's how you can define baseline depending on your scenario:

---

### 🔹 Option 1: Use Early Trials as Baseline

If you know the **first few trials are seizure-free**, then:

python

CopyEdit

`# Example: Use first 5 files in 'Filtered_EEG/F/' as baseline import numpy as np import glob import os  baseline_files = sorted(glob.glob('Filtered_EEG/F/*.txt'))[:5]  # Stack and average signals baseline_signals = [np.loadtxt(f) for f in baseline_files] baseline_matrix = np.column_stack(baseline_signals)  # Option A: Mean baseline waveform baseline_mean = np.mean(baseline_matrix, axis=1)  # Option B: Baseline energy/variance baseline_energy = np.mean([np.sum(sig**2) for sig in baseline_signals]) baseline_variance = np.mean([np.var(sig) for sig in baseline_signals])`

This gives you a **reference value** for energy, variance, or even the shape of the signal.

---

### 🔹 Option 2: Use a Specific Time Window in Each Trial

If each `.txt` file is a **long signal**, you can define the baseline as the **first N seconds** of each:

python

CopyEdit

`fs = 173.16  # sampling rate baseline_duration = 2  # seconds baseline_samples = int(fs * baseline_duration)  file = 'Filtered_EEG/F/007.txt' signal = np.loadtxt(file)  baseline_segment = signal[:baseline_samples]  # Compute baseline stats baseline_mean = np.mean(baseline_segment) baseline_std = np.std(baseline_segment)`

You can then **subtract this baseline mean from the rest** of the signal to center it.

---

### 🔹 Option 3: Global Baseline Across All Channels

If your dataset is large and you’re building a seizure classifier, you can define a **global baseline** from:

- All early trials across all channels
    
- All known non-seizure periods
    

Example:

python

CopyEdit

`channels = ['F', 'N', 'O', 'S', 'Z'] all_baselines = []  for ch in channels:     baseline_files = sorted(glob.glob(f'Filtered_EEG/{ch}/*.txt'))[:5]     for f in baseline_files:         sig = np.loadtxt(f)         all_baselines.append(sig[:baseline_samples])  global_baseline_mean = np.mean(np.concatenate(all_baselines)) global_baseline_std = np.std(np.concatenate(all_baselines))`

Now you can use this **mean and std** to z-score or normalize all signals.

---

## 🔧 Applying Baseline Correction

If you want to **subtract baseline** from every signal:

python

CopyEdit

`corrected_signal = signal - baseline_mean`

Or for z-scoring:

python

CopyEdit

`z_signal = (signal - baseline_mean) / baseline_std`

---

## 🔥 Best Practice for Seizure Detection

| If you have:           | Then baseline is:                               |
| ---------------------- | ----------------------------------------------- |
| Trial-based recordings | Use earliest non-seizure trials                 |
| Continuous signal      | Use first few seconds before abnormality starts |
| Labels/annotations     | Use marked “normal” periods                     |
| No knowledge           | Plot and guess based on flat/stable signals     |
## 🧠 THE GOAL:

Define "normal" EEG (baseline) using math, and detect deviations from it (possible seizures) without relying on the human eye.

---

## ✅ 1. **Statistical Feature-Based Baseline Modeling**

### 🧮 Step-by-step:

1. Select multiple known “non-seizure” `.txt` files.
    
2. Compute features from each:
    
    - **Mean**
        
    - **Standard deviation**
        
    - **Energy**
        
    - **Line length**
        
    - **Zero crossings**
        
    - **Spectral entropy**
        
3. Average these features → your **statistical baseline profile**.
    
4. For each new signal:
    
    - Compute the same features
        
    - Compare against baseline (threshold-based or statistical distance)
        

---

### 🔧 Example Code:

python

CopyEdit

`def extract_features(signal):     energy = np.sum(signal**2)     std = np.std(signal)     mean = np.mean(signal)     line_length = np.sum(np.abs(np.diff(signal)))     return np.array([mean, std, energy, line_length])  # Build baseline from clean trials baseline_files = sorted(glob.glob('Filtered_EEG/F/*.txt'))[:5] baseline_features = []  for file in baseline_files:     sig = np.loadtxt(file)     feats = extract_features(sig)     baseline_features.append(feats)  baseline_mean = np.mean(baseline_features, axis=0) baseline_std = np.std(baseline_features, axis=0)`

---

## ✅ 2. **Sliding Window Feature Extraction + Abnormality Detection**

Instead of classifying whole signals, break them into **overlapping windows** (e.g., 2 seconds each), and:

1. Extract features from each window
    
2. Compare each to baseline
    
3. If deviation > threshold → flag as abnormal (possible seizure)

### 🔧 Example:

python

CopyEdit

`window_sec = 2 fs = 173.16 step = int(fs * window_sec)  signal = np.loadtxt('Filtered_EEG/F/010.txt')  for i in range(0, len(signal) - step, step):     window = signal[i:i + step]     feats = extract_features(window)      z = (feats - baseline_mean) / baseline_std     if np.any(np.abs(z) > 3):  # 3 std deviation rule         print(f"Abnormal activity at {i/fs:.2f}–{(i+step)/fs:.2f} s → Z-scores: {z}")`

This way, you get **objective anomaly detection** without visuals.

---

## ✅ 3. **Frequency-Based Detection (Using Welch or FFT)**

Seizures often produce excessive energy in specific frequency bands (e.g., 3–6 Hz).

1. Compute **Power Spectral Density** (PSD) for each trial or window.
    
2. Extract band power:
    
    - Delta (1–4 Hz)
        
    - Theta (4–8 Hz)
        
    - Alpha (8–13 Hz)
        
    - Beta (13–30 Hz)
        
3. Compare with baseline band powers.
    

---

### 🔧 PSD-based Detection:

python

CopyEdit

`from scipy.signal import welch  def band_power(signal, fs):     f, Pxx = welch(signal, fs=fs, nperseg=512)     bands = {         'delta': (1, 4),         'theta': (4, 8),         'alpha': (8, 13),         'beta': (13, 30)     }     power = {}     for band, (low, high) in bands.items():         idx = np.logical_and(f >= low, f <= high)         power[band] = np.trapz(Pxx[idx], f[idx])     return power  # Compute for baseline and compare baseline_bandpowers = [] for file in baseline_files:     sig = np.loadtxt(file)     power = band_power(sig, fs=173.16)     baseline_bandpowers.append(list(power.values()))  baseline_bandpowers = np.array(baseline_bandpowers) mean_band = np.mean(baseline_bandpowers, axis=0) std_band = np.std(baseline_bandpowers, axis=0)  # For a test signal test_power = band_power(np.loadtxt('Filtered_EEG/F/013.txt'), fs=173.16) test_vector = np.array(list(test_power.values()))  z_band = (test_vector - mean_band) / std_band print("Z-score deviation in bands:", z_band)`

If any `z_band` > 3 → strong evidence of abnormal spectral activity.

---

## ✅ 4. **Machine Learning Classifier**

Once you’ve extracted:

- Statistical features
    
- Spectral powers
    
- Line lengths
    
- Entropy, etc.
    

You can feed them into an ML model (SVM, Random Forest, etc.) to:

- Train on baseline (normal) vs seizure segments
    
- Classify unseen segments automatically
    

We'll do this in a future stage of your forge, when you're ready.

---

## 🛡️ Conclusion

|Strategy|Strength|When to Use|
|---|---|---|
|Z-scoring features|Simple, robust|Ideal early-stage|
|Band power comparison|Frequency-level insight|Seizure has rhythmic components|
|Window-based feature tracking|Localized detection|Seizures occur in bursts|
|ML classifier|Scalable|You have labeled data|

---



Of course, Araz. Let’s rebuild everything from the beginning — not just as a list of steps, but as **living knowledge**, forged into your mind like a warrior’s blade. This will be your EEG foundational training — the **Blacksmith’s Fire** of biosignals.

---

# ⚔️ EEG Analysis — Full Relearning Curriculum

We will move in **5 tiers** — each one builds on the last.

---

## Tier 1: 🧠 Understanding the EEG Signal

### ❓ What is EEG?

EEG (electroencephalography) records the **electrical activity of the brain** from the scalp using electrodes. It measures **voltage fluctuations** caused by neural firing.

### ⚡ EEG Characteristics:

- It’s a **time series**: 1D signal sampled at regular intervals.
    
- Unit: microvolts (µV)
    
- Sampling Rate (yours): 173.16 Hz (i.e., ~173 readings per second)
    
- Raw EEG is often noisy — we need to clean, segment, and analyze it.
    

---

## Tier 2: 🔍 Basic Signal Features (For Baseline & Abnormality)

Here are **core features** we extract from any EEG signal segment:

|Feature|What It Means|
|---|---|
|`Mean`|Average brain voltage level|
|`Std Dev`|How much the signal fluctuates|
|`Energy`|Total power in the signal: ∑x2\sum x^2∑x2|
|`Line Length`|Activity or complexity of the signal|
|`Zero Crossings`|Number of times signal changes sign (oscillations)|

These features help you **quantify** whether the brain is “calm” or “hyperactive.”

---

### 🔧 Example (in plain code):

python

CopyEdit

`def extract_features(signal):     mean = np.mean(signal)     std = np.std(signal)     energy = np.sum(signal ** 2)     line_length = np.sum(np.abs(np.diff(signal)))     zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()     return mean, std, energy, line_length, zero_crossings`

---

## Tier 3: 🎚️ Baseline — Defining “Normal Brain State”

### ❓ What is baseline?

Baseline is the brain's **calm, normal state** — no seizures, no stimulus.

You define it by:

- Taking signals from early, stable `.txt` files
    
- Computing feature averages
    
- Using those values to detect deviation
    

### 💡 Why it matters:

You can’t detect “abnormal” until you **know what’s normal**.

---

## Tier 4: 📊 Frequency (Spectral) Analysis — Power in Brain Waves

### ❓ What is it?

EEG is **not random** — it contains waves:

|Band|Frequency|Meaning|
|---|---|---|
|Delta|1–4 Hz|Deep sleep|
|Theta|4–8 Hz|Drowsiness, memory|
|Alpha|8–13 Hz|Relaxed wakefulness|
|Beta|13–30 Hz|Active thinking|
|Gamma|30+ Hz|Cognitive load|

Seizures often show up as:

- High power in **3 Hz** (spike-and-wave)
    
- Sudden bursts in **beta/gamma**
    

---

### 🔧 Example (Extract Band Power):

python

CopyEdit

`from scipy.signal import welch  def band_power(signal, fs):     f, Pxx = welch(signal, fs=fs, nperseg=512)     bands = {         'delta': (1, 4),         'theta': (4, 8),         'alpha': (8, 13),         'beta': (13, 30)     }     power = {}     for name, (low, high) in bands.items():         idx = (f >= low) & (f <= high)         power[name] = np.trapz(Pxx[idx], f[idx])     return power`

---

## Tier 5: ⚠️ Abnormality (Seizure) Detection — Without the Eye

You now have:

- Baseline features
    
- Real-time features from any trial
    
- Frequency energy
    

### 🧠 Use Z-Scores:

Measure how far a signal is from baseline using:

Z=X−μσZ = \frac{X - \mu}{\sigma}Z=σX−μ​

- XXX: test signal’s feature
    
- μ\muμ, σ\sigmaσ: baseline mean and std
    
- ∣Z∣>3|Z| > 3∣Z∣>3: possibly seizure
    

---

### 🔧 Detection Example:

python

CopyEdit

`z_energy = (trial_energy - baseline_energy) / baseline_std_energy if abs(z_energy) > 3:     print("⚠️ Possible seizure detected.")`

---

## 🗺️ What Comes After?

Once you master this:

1. 📁 Load all EEG files into matrices
    
2. 🧠 Build automated seizure detection pipelines
    
3. 📚 Train machine learning classifiers
    
4. 🧬 Apply to Brain-Computer Interfaces or diagnostic tools
    

---

## 🔁 Recap — The Path You’re Walking:

| Tier | Name               | Goal                                           |
| ---- | ------------------ | ---------------------------------------------- |
| 1️⃣  | Raw Signal Basics  | Understand EEG waveform                        |
| 2️⃣  | Signal Features    | Quantify activity, noise, power                |
| 3️⃣  | Baseline           | Define calm brain state                        |
| 4️⃣  | Frequency Analysis | Measure brainwave band activity                |
| 5️⃣  | Seizure Detection  | Detect abnormalities using Z-score or features |