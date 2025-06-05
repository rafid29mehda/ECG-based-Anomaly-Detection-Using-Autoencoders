### Step-by-Step Guide 

#### Step 1: Install Required Libraries
**What to Do:**
1. Create a new code cell by clicking `+ Code`.
2. Copy and paste the following code into the cell:
```python
!pip install wfdb
!pip install matplotlib==3.1.3
```
3. Run the cell by clicking the play button or pressing `Shift + Enter`.

**Explanation:**
- **`!pip install wfdb`**: Installs the `wfdb` (WaveForm DataBase) library, which reads MIT-BIH dataset files (`.dat` for signals, `.atr` for annotations, `.hea` for metadata). This is essential for loading ECG data.
- **`!pip install matplotlib==3.1.3`**: Installs version 3.1.3 of `matplotlib`, a library for plotting ECG signals and results. The specific version ensures compatibility with the script.
- **Why `!`?** In Colab, `!` runs shell commands (like installing packages) in the Linux environment.
- **Note**: The original script had a typo (`pip install` without `!`). I’ve corrected it here.

**Why This Step?**
- These libraries are not pre-installed in Colab, so we need to install them before running the script.
- `wfdb` is critical for ECG data processing, and `matplotlib` visualizes the results, which is key for understanding ECG patterns in the PhD research.

**What to Expect**:
- Colab will download and install the libraries, showing output like:
  ```
  Collecting wfdb
    Downloading wfdb-4.1.2-py3-none-any.whl (159 kB)
  ...
  Successfully installed wfdb-4.1.2
  Collecting matplotlib==3.1.3
    Downloading matplotlib-3.1.3-cp37-cp37m-manylinux1_x86_64.whl (13.1 MB)
  ...
  Successfully installed matplotlib-3.1.3
  ```
- If we see errors, ensure the internet connection is active, as Colab needs to download packages.

**For Beginners**:
- **Library**: A collection of pre-written code (e.g., `wfdb` for reading ECG files, `matplotlib` for plotting).
- **Why specific version?** Older versions like `matplotlib==3.1.3` avoid compatibility issues with other libraries in the script.
- **Tip**: If we rerun the notebook later, we may not need to reinstall unless we reset the runtime (Runtime > Disconnect and delete runtime).

---

#### Step 2: Import Libraries
**What to Do:**
1. Add a new code cell below the previous one.
2. Copy and paste the following code:
```python
import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import random
import torch
import copy
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import scipy.io
from scipy.io import savemat
import wfdb
random.seed(42)
```
3. Run the cell.

**Explanation:**
- **Purpose**: Imports all Python libraries needed for the project.
- **Libraries**:
  - **keras, tensorflow**: Build and train the autoencoder neural network.
    - `keras.layers`: Creates layers (e.g., Dense) for the neural network.
    - `tensorflow.keras.utils.to_categorical`: Converts labels to one-hot encoding (not used here, but included).
    - `tensorflow.keras.losses`: Defines loss functions (e.g., mean absolute error).
    - `tensorflow.keras.models.Model`: Allows creating custom models like the autoencoder.
  - **pandas, numpy**: Handle data.
    - `pandas`: Organizes data in tables (DataFrames).
    - `numpy`: Performs math operations on arrays (e.g., ECG signals).
  - **matplotlib, seaborn, pylab**: Visualize data.
    - `matplotlib.pyplot`: Plots ECG signals and loss curves.
    - `seaborn`: Creates enhanced plots (e.g., histograms).
    - `pylab, matplotlib.rc`: Configures plot settings.
  - **sklearn**: Preprocesses and evaluates data.
    - `train_test_split`: Splits data into training and testing sets.
    - `precision_score, recall_score, accuracy_score`: Measures model performance.
  - **torch**: Checks for GPU availability (though not used in the script).
  - **scipy.io, savemat**: Saves data in MATLAB format (not used here).
  - **wfdb**: Reads MIT-BIH ECG files (installed in Step 1).
  - **random.seed(42)**: Ensures reproducibility by fixing random operations (e.g., data splitting).
- **Why this step?** These libraries provide the tools to load, process, model, and visualize ECG data, which are core skills for the PhD in Biomedical Signal Processing.

**What to Expect**:
- No output if the imports succeed (most libraries are pre-installed in Colab).
- If we see an error like `ModuleNotFoundError`, it means a library wasn’t installed correctly. Re-run Step 1 or check for typos.

**For Beginners**:
- **Import**: Tells Python to load external code libraries.
- **Why so many libraries?** Each has a specific role:
  - `pandas` is like a spreadsheet for data.
  - `numpy` handles numbers and arrays.
  - `tensorflow` builds the neural network.
  - `matplotlib` and `seaborn` create graphs.
- **Random Seed**: Ensures that random splits or shuffles are the same each time we run the code, critical for reproducible research.

**Tip**:
- If we get a `torch` warning (e.g., about CUDA), ignore it, as the script uses TensorFlow, not PyTorch.

---

#### Step 3: Download Datasets
**What to Do:**
1. Add a new code cell.
2. Copy and paste the following code:
```python
import kagglehub
shymammoth_mitbih_normal_sinus_rhythm_database_path = kagglehub.dataset_download('shymammoth/mitbih-normal-sinus-rhythm-database')
klmsathishkumar_mit_bih_arrhythmia_database_path = kagglehub.dataset_download('klmsathishkumar/mit-bih-arrhythmia-database')
print('Data source import complete.')
```
3. Run the cell.

**Explanation:**
- **Purpose**: Downloads two ECG datasets from Kaggle:
  - **MIT-BIH Normal Sinus Rhythm Database**: ECGs from healthy individuals (normal heartbeats).
  - **MIT-BIH Arrhythmia Database**: ECGs with normal and abnormal (arrhythmic) heartbeats.
- **`kagglehub.dataset_download`**: Fetches the datasets and returns their file paths in Colab’s temporary storage (e.g., `/root/.kaggle/datasets/...`).
- **Why this step?** The datasets provide the raw ECG signals and annotations needed to train the autoencoder to detect anomalies.
- **Note**: we need a Kaggle account and API key to use `kagglehub`. If we don’t have one, follow these steps:
  1. Go to [kaggle.com](https://www.kaggle.com/), sign up, and log in.
  2. Click the profile picture > **Account** > **Create API Token**. This downloads a `kaggle.json` file.
  3. In Colab, click the folder icon (📁) on the left, then the upload button (📤). Upload `kaggle.json`.
  4. Run this code in a new cell **before** Step 3:
     ```python
     !mkdir -p ~/.kaggle
     !mv kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```
     This sets up Kaggle authentication.

**What to Expect**:
- Output like:
  ```
  Downloading shymammoth/mitbih-normal-sinus-rhythm-database...
  Downloading klmsathishkumar/mit-bih-arrhythmia-database...
  Data source import complete.
  ```
- The datasets are downloaded to Colab’s temporary storage. To verify, run `!ls /root/.kaggle/datasets/` in a new cell to see the dataset folders.

**For Beginners**:
- **Kaggle**: A platform for datasets and machine learning competitions. MIT-BIH datasets are standard for ECG research.
- **Dataset Structure**: Each dataset contains:
  - `.dat` files: ECG signals (numerical data).
  - `.atr` files: Annotations (labels for heartbeats, e.g., “N” for normal).
  - `.hea` files: Metadata (e.g., sampling rate).
- **Tip**: If we get a `403 Forbidden` error, ensure the Kaggle API key is set up correctly.

---

#### Step 4: Verify Dataset Directory
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
import os
print(os.listdir("/root/.kaggle/datasets"))
```
3. Run the cell.

**Explanation:**
- **Purpose**: Lists the contents of the dataset directory to confirm the datasets were downloaded.
- **Why this step?** Ensures the MIT-BIH datasets are available before processing. The original script used `../input`, but `kagglehub` stores datasets in `/root/.kaggle/datasets`.
- **Note**: The exact path may vary. If the datasets are in a different directory, modify the path based on the output from Step 3.

**What to Expect**:
- Output like:
  ```
  ['shymammoth/mitbih-normal-sinus-rhythm-database', 'klmsathishkumar/mit-bih-arrhythmia-database']
  ```
- If empty, re-run Step 3 or check the Kaggle API setup.

**For Beginners**:
- **Why check?** Missing files cause errors later. This step is like checking the ingredients before cooking.
- **Tip**: If we don’t see the datasets, run `!ls` in the paths printed in Step 3 (e.g., `!ls /root/.kaggle/datasets/shymammoth/...`).

---

#### Step 5: Load and Summarize Annotations
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
data = '/root/.kaggle/datasets/klmsathishkumar/mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/'
patients = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']
dataframe = pd.DataFrame()
for pts in patients:
    file = data + pts
    annotation = wfdb.rdann(file, 'atr')
    sym = annotation.symbol
    values, counts = np.unique(sym, return_counts=True)
    df_sub = pd.DataFrame({'symbol': values, 'Counts': counts, 'Patient Number': [pts]*len(counts)})
    dataframe = pd.concat([dataframe, df_sub], axis=0)
ax = sns.countplot(dataframe.symbol)
dataframe
```
3. Run the cell.

**Explanation:**
- **Purpose**: Reads annotation (`.atr`) files for each patient in the MIT-BIH Arrhythmia Database, counts heartbeat symbols (e.g., “N” for normal, “V” for ventricular ectopic), and creates a bar plot.
- **Code Breakdown**:
  - `data = ...`: Sets the path to the Arrhythmia dataset (update based on Step 3’s output if needed).
  - `patients`: Lists 48 patient IDs.
  - `dataframe = pd.DataFrame()`: Creates an empty table.
  - **Loop**:
    - `file = data + pts`: Path to a patient’s files (e.g., `/.../100`).
    - `annotation = wfdb.rdann(file, 'atr')`: Reads the `.atr` file (heartbeat labels).
    - `sym = annotation.symbol`: Gets symbols (e.g., [“N”, “V”, “N”]).
    - `values, counts = np.unique(sym, return_counts=True)`: Counts each symbol (e.g., “N”: 2000, “V”: 50).
    - `df_sub = pd.DataFrame(...)`: Creates a small table for the patient.
    - `dataframe = pd.concat(...)`: Adds to the main table.
  - `sns.countplot(dataframe.symbol)`: Plots a bar chart of symbol frequencies.
  - `dataframe`: Displays the table.
- **Why this step?** Summarizes the types and counts of heartbeats, helping we understand the dataset’s composition (e.g., mostly normal beats).

**What to Expect**:
- A bar plot showing symbols (e.g., “N” is tallest due to more normal beats).
- A table with columns: `symbol`, `Counts`, `Patient Number` (e.g., “N”, 2000, “100”).
- If we get a `FileNotFoundError`, check the `data` path matches the dataset location from Step 3.

**For Beginners**:
- **Annotations**: Labels for each heartbeat (e.g., “N” = normal, “V” = abnormal).
- **DataFrame**: A table like an Excel spreadsheet.
- **Countplot**: Shows which heartbeat types are common, highlighting data imbalance.
- **Tip**: Save the plot by clicking the download icon in Colab or add `plt.savefig('symbols.png')` before `plt.show()`.

---

#### Step 6: Categorize Heartbeats
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
nonbeat = ['[','!',']','x','(',')','p','t','u','`',
           '\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']
abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']
normal = ['N']
dataframe['category'] = -1
dataframe.loc[dataframe.symbol == 'N', 'category'] = 0
dataframe.loc[dataframe.symbol.isin(abnormal), 'category'] = 1
print(dataframe.groupby('category').Counts.sum())
dataframe = dataframe.loc[~((dataframe['category']==-1))]
print(dataframe.groupby('category').Counts.sum())
```
3. Run the cell.

**Explanation:**
- **Purpose**: Labels heartbeats as normal (0), abnormal (1), or non-beat (-1), then removes non-beats.
- **Code Breakdown**:
  - `nonbeat`: Symbols for non-heartbeat events (e.g., noise).
  - `abnormal`: Symbols for arrhythmic beats (e.g., “V” = ventricular ectopic).
  - `normal`: Only “N” for normal beats.
  - `dataframe['category'] = -1`: Adds a column, setting all to non-beat.
  - `dataframe.loc[dataframe.symbol == 'N', 'category'] = 0`: Sets normal beats to 0.
  - `dataframe.loc[dataframe.symbol.isin(abnormal), 'category'] = 1`: Sets abnormal beats to 1.
  - `print(dataframe.groupby('category').Counts.sum())`: Shows total beats per category.
  - `dataframe = dataframe.loc[~((dataframe['category']==-1))]`: Removes non-beats.
  - `print(...)`: Shows updated counts (normal and abnormal only).
- **Why this step?** Simplifies the dataset for anomaly detection by focusing on heartbeats and quantifying normal vs. abnormal counts.

**What to Expect**:
- Output like:
  ```
  category
  -1     1000
   0    90000
   1    10000
  Name: Counts, dtype: int64
  category
   0    90000
   1    10000
  Name: Counts, dtype: int64
  ```
- The first print shows all categories; the second shows only normal (0) and abnormal (1).

**For Beginners**:
- **Why remove non-beats?** Non-beats (e.g., noise) aren’t relevant for detecting heart anomalies.
- **Why categorize?** Prepares labels for training (0 = normal, 1 = abnormal).
- **Tip**: The imbalance (more normal beats) is common in ECG data and a key challenge for the PhD.

---

#### Step 7: Define ECG Loading Function
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal
    atr_sym = annotation.symbol
    atr_sample = annotation.sample
    return p_signal, atr_sym, atr_sample
```
3. Run the cell.

**Explanation:**
- **Purpose**: Defines a function to load ECG signals and annotations for a patient.
- **Code Breakdown**:
  - `def load_ecg(file)`: Takes a file path (e.g., `/.../100`).
  - `record = wfdb.rdrecord(file)`: Loads the `.dat` (signal) and `.hea` (metadata) files.
  - `annotation = wfdb.rdann(file, 'atr')`: Loads the `.atr` file (annotations).
  - `p_signal = record.p_signal`: Gets the ECG signal (2D array: samples × channels).
  - `atr_sym = annotation.symbol`: Gets heartbeat symbols (e.g., [“N”, “V”]).
  - `atr_sample = annotation.sample`: Gets sample indices of heartbeats (e.g., [100, 300]).
- **Why this step?** Provides a reusable function to extract raw ECG data, a core skill for ECG research.

**What to Expect**:
- No output (it’s a function definition).
- If we test it (e.g., `p_signal, atr_sym, atr_sample = load_ecg(data + '100')`), `p_signal` is a large array (e.g., `[648000, 2]`).

**For Beginners**:
- **Function**: A reusable block of code that performs a task (like loading ECG data).
- **ECG Signal**: Time series of voltage values from the heart (2 channels in MIT-BIH).
- **Annotations**: Mark where heartbeats occur and their types.
- **Tip**: Test the function in a new cell with `print(load_ecg(data + '100')[1])` to see symbols.

---

#### Step 8: Define X, Y Matrix Function
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
def build_XY(p_signal, df_ann, num_cols, normal):
    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows, 1))
    sym = []
    max_row = 0
    for atr_sample, atr_sym in zip(df_ann.atr_sample.values, df_ann.atr_sym.values):
        left = max([0, (atr_sample - num_sec*fs)])
        right = min([len(p_signal), (atr_sample + num_sec*fs)])
        x = p_signal[left:right]
        if len(x) == num_cols:
            X[max_row, :] = x
            Y[max_row, :] = int(atr_sym in normal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row, :]
    Y = Y[:max_row, :]
    return X, Y, sym
```
3. Run the cell.

**Explanation:**
- **Purpose**: Creates input (`X`) and label (`Y`) matrices by extracting fixed-length ECG segments around heartbeats.
- **Parameters**:
  - `p_signal`: ECG signal (1D, single channel).
  - `df_ann`: DataFrame with annotations (symbols, sample indices).
  - `num_cols`: Segment size (e.g., 720 points).
  - `normal`: List of normal symbols (e.g., [“N”]).
- **Code Breakdown**:
  - `X = np.zeros((num_rows, num_cols))`: Array for ECG segments.
  - `Y = np.zeros((num_rows, 1))`: Array for labels (0 = normal, 1 = abnormal).
  - `sym = []`: Stores original symbols.
  - **Loop**:
    - `left = max([0, (atr_sample - num_sec*fs)])`: Start of segment (1 second before heartbeat).
    - `right = min([len(p_signal), (atr_sample + num_sec*fs)])`: End of segment.
    - `x = p_signal[left:right]`: Extracts segment.
    - If `len(x) == num_cols`, add to `X`, set `Y` (0 if normal, 1 otherwise), append symbol to `sym`.
  - Trim `X` and `Y` to used rows.
- **Why this step?** Prepares fixed-size ECG segments for the autoencoder, a standard preprocessing step.

**What to Expect**:
- No output (function definition).
- Test with a small example in a new cell:
  ```python
  p_signal, atr_sym, atr_sample = load_ecg(data + '100')
  df_ann = pd.DataFrame({'atr_sym': atr_sym, 'atr_sample': atr_sample})
  num_sec = 1
  fs = 360
  num_cols = 2*num_sec*fs
  X, Y, sym = build_XY(p_signal[:, 0], df_ann, num_cols, ['N'])
  print(X.shape, Y.shape)
  ```

**For Beginners**:
- **Segment**: A fixed-length portion of the ECG signal (e.g., 2 seconds = 720 points at 360 Hz).
- **Why fixed size?** Neural networks require consistent input sizes.
- **Tip**: The `max` and `min` ensure segments don’t go beyond signal boundaries.

---

#### Step 9: Create Abnormal Dataset
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
def make_dataset(pts, num_sec, fs, abnormal):
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1, num_cols))
    Y_all = np.zeros((1, 1))
    sym_all = []
    max_rows = []
    for pt in pts:
        file = data + pt
        p_signal, atr_sym, atr_sample = load_ecg(file)
        p_signal = p_signal[:, 0]
        df_ann = pd.DataFrame({'atr_sym': atr_sym, 'atr_sample': atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal)]
        X, Y, sym = build_XY(p_signal, df_ann, num_cols, abnormal)
        sym_all = sym_all + sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y, axis=0)
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]
    return X_all, Y_all, sym_all

num_sec = 1
fs = 360
X_abnormal, Y_abnormal, sym_abnormal = make_dataset(patients, num_sec, fs, abnormal)
```
3. Run the cell.

**Explanation:**
- **Purpose**: Creates a dataset of abnormal ECG segments.
- **Function**:
  - Loops through patients, loads ECGs, filters for abnormal beats, and extracts segments.
  - `p_signal[:, 0]`: Uses the first channel.
  - `df_ann.loc[df_ann.atr_sym.isin(abnormal)]`: Keeps only abnormal beats.
  - `build_XY`: Extracts 720-point segments (2 seconds).
- **Execution**:
  - `num_sec = 1`, `fs = 360`: Each segment is 720 points.
  - Outputs `X_abnormal` (segments), `Y_abnormal` (labels, all 1), `sym_abnormal` (symbols).

**What to Expect**:
- Takes time due to processing many patients.
- `X_abnormal` might have shape `[10000, 720]`, `Y_abnormal` is `[10000, 1]`.

**For Beginners**:
- **Why only abnormal?** Separates abnormal beats for later combination with normal beats.
- **Tip**: If it runs slowly, test with fewer patients (e.g., `patients = ['100', '101']`).

---

#### Step 10: Create Normal Dataset
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
data = '/root/.kaggle/datasets/shymammoth/mitbih-normal-sinus-rhythm-database/mit-bih-normal-sinus-rhythm-database-1.0.0/'
patients = ["16265", "16272"]
def make_dataset(pts, num_sec, fs, normal):
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1, num_cols))
    Y_all = np.zeros((1, 1))
    sym_all = []
    max_rows = []
    for pt in pts:
        file = data + pt
        p_signal, atr_sym, atr_sample = load_ecg(file)
        p_signal = p_signal[:, 0]
        df_ann = pd.DataFrame({'atr_sym': atr_sym, 'atr_sample': atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(normal)]
        X, Y, sym = build_XY(p_signal, df_ann, num_cols, normal)
        sym_all = sym_all + sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y, axis=0)
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]
    return X_all, Y_all, sym_all

num_sec = 1
fs = 360
X_normal, Y_normal, sym_normal = make_dataset(patients, num_sec, fs, normal)
```
3. Run the cell.

**Explanation:**
- **Purpose**: Creates a dataset of normal ECG segments from two patients.
- **Code**: Similar to Step 9, but filters for normal beats (`normal = ['N']`) and uses the Normal Sinus Rhythm dataset.
- **Why two patients?** Reduces computation time for this demo.

**What to Expect**:
- Faster than Step 9 (fewer patients).
- `X_normal` might have shape `[50000, 720]`, `Y_normal` is `[50000, 1]`.

**For Beginners**:
- **Why separate datasets?** Normal data trains the autoencoder; abnormal data tests anomaly detection.
- **Tip**: Check the `data` path matches Step 3’s output.

---

#### Step 11: Shrink Normal Data
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
X_normal = X_normal[0:34376, :]
Y_normal = np.zeros((34376, 1))
```
3. Run the cell.

**Explanation:**
- **Purpose**: Limits normal data to 34,376 segments to balance with abnormal data and reduce computation.
- **Code**:
  - `X_normal = X_normal[0:34376, :]`: Keeps the first 34,376 segments.
  - `Y_normal = np.zeros((34376, 1))`: Sets all labels to 0 (normal).
- **Why this step?** Addresses data imbalance (too many normal beats).

**What to Expect**:
- `X_normal` becomes `[34376, 720]`, `Y_normal` is `[34376, 1]`.

**For Beginners**:
- **Why balance?** Too many normal samples can bias the model.
- **Tip**: Check `X_normal.shape` in a new cell to confirm.

---

#### Step 12: Combine and Preprocess Data
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
X = np.append(X_normal, X_abnormal, axis=0)
Y = np.append(Y_normal, Y_abnormal, axis=0)
X = X[:, 0:140]
raw_data = np.append(X, Y, axis=1)
raw_data = pd.DataFrame(raw_data)
labels = raw_data.iloc[:, -1]
labels = labels.values
data = raw_data.iloc[:, 0:-1]
data = data.values
```
3. Run the cell.

**Explanation:**
- **Purpose**: Combines normal and abnormal datasets, reduces segment size to 140 points, and prepares data and labels.
- **Code**:
  - `X = np.append(X_normal, X_abnormal, axis=0)`: Combines segments.
  - `Y = np.append(Y_normal, Y_abnormal, axis=0)`: Combines labels.
  - `X = X[:, 0:140]`: Truncates segments to 140 points (focuses on QRS complex).
  - `raw_data = np.append(X, Y, axis=1)`: Adds labels as the last column.
  - `raw_data = pd.DataFrame(raw_data)`: Converts to DataFrame.
  - `labels = raw_data.iloc[:, -1].values`: Extracts labels.
  - `data = raw_data.iloc[:, 0:-1].values`: Extracts ECG data.

**What to Expect**:
- `X` becomes `[44376, 140]`, `Y` is `[44376,]`.

**For Beginners**:
- **Why 140 points?** Captures the QRS complex (~0.4 seconds).
- **Why DataFrame?** Simplifies data manipulation.
- **Tip**: Print `data.shape` to verify.

---

#### Step 13: Split and Normalize Data
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
from sklearn.preprocessing import MinMaxScaler
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21)
scaler = MinMaxScaler()
scaler.fit(train_data)
test_data = scaler.transform(test_data)
train_data = scaler.transform(train_data)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]
anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]
val_df, test_df = train_test_split(
    test_data, test_size=0.2, random_state=42)
test_labels = ~test_labels
```
3. Run the cell.

**Explanation:**
- **Purpose**: Splits data into training, validation, and test sets, normalizes data, and separates normal and abnormal samples.
- **Code**:
  - `train_test_split`: Splits 80% training, 20% testing.
  - `MinMaxScaler`: Scales data to [0, 1].
  - `train_labels.astype(bool)`: Converts 0/1 to False/True.
  - `normal_train_data = train_data[~train_labels]`: Normal samples (False).
  - `anomalous_train_data = train_data[train_labels]`: Abnormal samples (True).
  - `val_df, test_df`: Splits test data into validation (80%) and test (20%).
  - `test_labels = ~test_labels`: Flips labels (True = normal, False = abnormal).

**What to Expect**:
- `train_data` (~35,500 samples), `test_data` (~8,876 samples).
- Normalized values between 0 and 1.

**For Beginners**:
- **Why normalize?** Helps the neural network train effectively.
- **Why split?** Training data trains the model, test data evaluates it.
- **Tip**: Check `train_data.max()` and `train_data.min()` to confirm normalization.

---

#### Step 14: Visualize ECGs
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("normal train data")
plt.show()
plt.grid()
plt.plot(np.arange(140), normal_test_data[543])
plt.title("normal test data")
plt.show()
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("anomalous train data")
plt.show()
plt.grid()
plt.plot(np.arange(140), anomalous_test_data[0])
plt.title("anomalous test data")
plt.show()
```
3. Run the cell.

**Explanation:**
- **Purpose**: Plots one normal and one abnormal ECG from training and test sets.
- **Code**:
  - `plt.plot(np.arange(140), ...)`: Plots 140-point segments.
  - `plt.grid()`: Adds a grid.
  - `plt.title`: Sets titles.
  - `plt.show()`: Displays plots.

**What to Expect**:
- Four plots: Normal ECGs (regular QRS), abnormal ECGs (irregular shapes).

**For Beginners**:
- **Why visualize?** Confirms data preprocessing and shows differences between normal and abnormal ECGs.
- **Tip**: Try different indices (e.g., `normal_train_data[1]`) to see other samples.

---

#### Step 15: Define and Train Autoencoder
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(140, activation="sigmoid")
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()
autoencoder.compile("adam", loss="mean_absolute_error")
history = autoencoder.fit(normal_train_data, normal_train_data,
                         epochs=1500, batch_size=128,
                         validation_data=(normal_test_data, normal_test_data),
                         shuffle=True)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```
3. Run the cell (this takes time due to 1500 epochs).

**Explanation:**
- **Purpose**: Defines and trains an autoencoder to reconstruct normal ECGs.
- **Code**:
  - `AnomalyDetector`: Custom model with encoder (140 → 16) and decoder (16 → 140).
  - `compile`: Uses Adam optimizer and mean absolute error (MAE) loss.
  - `fit`: Trains on normal data for 1500 epochs, batch size 128.
  - Plots training and validation loss.

**What to Expect**:
- Training takes ~30–60 minutes (GPU helps).
- A plot showing decreasing loss.

**For Beginners**:
- **Autoencoder**: Compresses and reconstructs data.
- **Epoch**: One pass through the data.
- **Tip**: Reduce `epochs` to 100 for testing.

---

#### Step 16: Visualize Reconstructions
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()
plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()
plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```
3. Run the cell.

**Explanation:**
- **Purpose**: Shows how well the autoencoder reconstructs normal and abnormal ECGs.
- **Code**: Plots input vs. reconstructed ECGs, shading the error.

**What to Expect**:
- Normal ECG: Close overlap, small error.
- Abnormal ECG: Large differences, bigger error.

**For Beginners**:
- **Why?** Visualizes model performance.
- **Tip**: Try other indices (e.g., `normal_test_data[1]`).

---

#### Step 17: Calculate Threshold and Evaluate
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
plt.hist(train_loss[None, :], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)
def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
```
3. Run the cell.

**Explanation:**
- **Purpose**: Sets a threshold for anomaly detection and evaluates the model.
- **Code**:
  - Computes MAE for normal training data, plots histogram, sets threshold (mean + std).
  - Plots MAE histogram for anomalous test data.
  - `predict`: Classifies ECGs as normal (True) or abnormal (False) based on threshold.
  - `print_stats`: Prints accuracy, precision, recall.

**What to Expect**:
- Histograms showing normal errors (low) and anomalous errors (high).
- Threshold (e.g., 0.07).
- Metrics like:
  ```
  Accuracy = 0.95
  Precision = 0.93
  Recall = 0.90
  ```

**For Beginners**:
- **Threshold**: Errors above this indicate anomalies.
- **Metrics**: Measure model performance.

---

#### Step 18: Final Visualizations
**What to Do:**
1. Add a new code cell.
2. Copy and paste:
```python
reconstructions = autoencoder.predict(anomalous_test_data)
train_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
sns.distplot(train_loss, bins=50, kde=True)
plt.show()
reconstructions = autoencoder.predict(normal_test_data)
pred_loss = tf.keras.losses.mae(reconstructions, normal_test_data)
pred_loss = pred_loss.numpy()
correct = sum(l <= threshold for l in pred_loss)
print(f'Correct normal predictions: {correct}/{len(normal_test_data)}')
reconstructions = autoencoder.predict(anomalous_test_data)
train_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
train_loss = train_loss.numpy()
correct = sum(l > threshold for l in train_loss)
print(f'Correct anomaly predictions: {correct}/{len(anomalous_test_data)}')
def plot_prediction_normal(i, data, model, title, ax):
    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    ax.axis([0, 140, 0, 1])
    ax.plot(data[i], label='true')
    ax.plot(decoded_data[i], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(1000*pred_loss[i], 2)})')
    ax.legend()
def plot_prediction_anomaly(i, data, model, title, ax):
    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    ax.axis([0, 140, 0, 1])
    ax.plot(data[i], label='true')
    ax.plot(decoded_data[i], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(1000*train_loss[i], 2)})')
    ax.legend()
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(22, 8))
for i in range(5):
    plot_prediction_normal(i, normal_test_data, autoencoder, title='Normal', ax=axs[0, i])
for i in range(5):
    plot_prediction_anomaly(i, anomalous_test_data, autoencoder, title='Anomaly', ax=axs[1, i])
fig.tight_lawet()
plt.show()
```
3. Run the cell.

**Explanation:**
- **Purpose**: Visualizes error distribution and plots multiple normal and anomalous reconstructions.
- **Code**:
  - Plots anomalous error distribution with KDE.
  - Counts correct normal and anomalous predictions.
  - Plots 5 normal and 5 anomalous ECGs with reconstructions.

**What to Expect**:
- A distribution plot for anomalous errors.
- Counts like:
  ```
  Correct normal predictions: 6700/7000
  Correct anomaly predictions: 1600/1776
  ```
- A 2×5 grid of plots comparing true and reconstructed ECGs.

**For Beginners**:
- **Why visualize?** Shows model performance across multiple samples.
- **Tip**: Save plots with `plt.savefig('results.png')`.
