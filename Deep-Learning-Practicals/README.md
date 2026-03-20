# Deep Learning Practicals

This folder contains practical implementations for ANN, CNN, RNN, LSTM, and GRU using TensorFlow and/or PyTorch.

## Setup

Install required packages:

```bash
pip install -r requirements.txt
```

## Folder Structure

```text
Deep-Learning-Practicals/
|- requirements.txt
|- README.md
|- ANN/
|  |- data/
|  |  |- Churn_Modelling.csv
|  |- PyTorch/
|  |  |- practical.ipynb
|  |- Tensorflow/
|     |- practical.ipynb
|- CNN/
|  |- Dataset/
|  |  |- 0/
|  |  |- 1/
|  |  |- 2/
|  |  |- 3/
|  |  |- 4/
|  |  |- 5/
|  |  |- 6/
|  |  |- 7/
|  |  |- 8/
|  |  |- 9/
|  |- PyTorch/
|  |  |- Sign_language_digit_classification.ipynb
|  |  |- Data/
|  |  |  |- train/
|  |  |  |  |- 0/
|  |  |  |  |- 1/
|  |  |  |  |- 2/
|  |  |  |  |- 3/
|  |  |  |  |- 4/
|  |  |  |  |- 5/
|  |  |  |  |- 6/
|  |  |  |  |- 7/
|  |  |  |  |- 8/
|  |  |  |  |- 9/
|  |  |  |- test/
|  |  |     |- 0/
|  |  |     |- 1/
|  |  |     |- 2/
|  |  |     |- 3/
|  |  |     |- 4/
|  |  |     |- 5/
|  |  |     |- 6/
|  |  |     |- 7/
|  |  |     |- 8/
|  |  |     |- 9/
|  |  |- Model_Performance/
|  |  |- Models/
|  |     |- sign_lan_digit_model.pth
|  |- Tensorflow/
|     |- Sign_language_digit_classification.ipynb
|     |- Model_Performance/
|     |- Models/
|        |- sign_lan_digit_model.h5
|- GRU/
|  |- Pytorch/
|     |- weather-forecasting-daily.ipynb
|     |- temp_avg/
|        |- naive_gru_model.pth
|- LSTM/
|  |- PyTorch/
|  |  |- weather-forecasting-daily.ipynb
|  |  |- temp_avg/
|  |     |- naive_model.pth
|  |- Tensorflow/
|     |- weather-forecasting-dailyc28e331197.ipynb
|     |- temp_avg/
|     |  |- temperature_avg_Standard.keras
|     |- temp_max/
|     |  |- temperature_max_Done5_Standard.keras
|     |- temp_min/
|        |- temperature_min_Standard.keras
|- RNN/
   |- pytorch_imple.ipynb
   |- Dataset/
   |  |- imdb/
   |     |- dataset_dict.json
   |     |- plain_text/0.0.0/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/
   |     |  |- dataset_info.json
   |     |  |- imdb-test.arrow
   |     |  |- imdb-train.arrow
   |     |  |- imdb-unsupervised.arrow
   |     |- train/
   |     |  |- data-00000-of-00001.arrow
   |     |  |- dataset_info.json
   |     |  |- state.json
   |     |- test/
   |     |  |- data-00000-of-00001.arrow
   |     |  |- dataset_info.json
   |     |  |- state.json
   |     |- unsupervised/
   |        |- data-00000-of-00001.arrow
   |        |- dataset_info.json
   |        |- state.json
   |- Models/
```

## Data Used

### 1. ANN
- File: `ANN/data/Churn_Modelling.csv`
- Type: tabular customer churn dataset
- Used in:
  - `ANN/Tensorflow/practical.ipynb`
  - `ANN/PyTorch/practical.ipynb`

### 2. CNN
- Source folder: `CNN/Dataset/`
- Classes: `0` to `9` (sign language digits)
- Processed/train-test split used in:
  - `CNN/PyTorch/Data/train/0..9`
  - `CNN/PyTorch/Data/test/0..9`
- Used in:
  - `CNN/PyTorch/Sign_language_digit_classification.ipynb`
  - `CNN/Tensorflow/Sign_language_digit_classification.ipynb`

### 3. RNN (Sentiment Analysis)
- Dataset: IMDB reviews
- Local path: `RNN/Dataset/imdb/`
- Storage format: Hugging Face Arrow files (`train`, `test`, `unsupervised`)
- Used in:
  - `RNN/pytorch_imple.ipynb`

### 4. LSTM (Weather Forecasting)
- Dataset source: Meteostat daily weather data (API fetched in notebook)
- Model artifacts:
  - `LSTM/PyTorch/temp_avg/naive_model.pth`
  - `LSTM/Tensorflow/temp_avg/temperature_avg_Standard.keras`
  - `LSTM/Tensorflow/temp_max/temperature_max_Done5_Standard.keras`
  - `LSTM/Tensorflow/temp_min/temperature_min_Standard.keras`
- Used in:
  - `LSTM/PyTorch/weather-forecasting-daily.ipynb`
  - `LSTM/Tensorflow/weather-forecasting-dailyc28e331197.ipynb`

### 5. GRU (Weather Forecasting)
- Dataset source: Meteostat daily weather data (API fetched in notebook)
- Model artifact:
  - `GRU/Pytorch/temp_avg/naive_gru_model.pth`
- Used in:
  - `GRU/Pytorch/weather-forecasting-daily.ipynb`

## Outputs and Saved Models

- ANN:
  - Notebook outputs inside notebook cells
- CNN:
  - `CNN/PyTorch/Models/sign_lan_digit_model.pth`
  - `CNN/Tensorflow/Models/sign_lan_digit_model.h5`
  - Performance plots/reports in `Model_Performance/`
- LSTM:
  - PyTorch and TensorFlow model files in `temp_*` folders
- GRU:
  - `naive_gru_model.pth` in `GRU/Pytorch/temp_avg/`
- RNN:
  - Dataset cached in `RNN/Dataset/imdb/`
  - `RNN/Models/` reserved for saved checkpoints
