# Data Folder

This folder is intentionally left empty — no raw or processed data files are included in this repository.

## Dataset Source

The dataset used in this project can be downloaded from Figshare:

**Title:** Predicting energy cost from wearable sensors: A dataset of energetic and physiological wearable sensor data from healthy individuals performing multiple physical activities  
**Authors:** Kimberly A. Ingraham, Daniel P. Ferris, and C. David Remy  
**DOI:** [https://figshare.com/articles/dataset/Predicting_energy_cost_from_wearable_sensors_A_dataset_of_energetic_and_physiological_wearable_sensor_data_from_healthy_individuals_performing_multiple_physical_activities/7473191](https://figshare.com/articles/dataset/Predicting_energy_cost_from_wearable_sensors_A_dataset_of_energetic_and_physiological_wearable_sensor_data_from_healthy_individuals_performing_multiple_physical_activities/7473191)

## ⚙️ Data Preparation

The Figshare dataset contains raw sensor data.  
Before using it, the data should be **cleaned and preprocessed** as described in the original paper:

> *Ingraham, K. A., Ferris, D. P., & Remy, C. D. (2019). Evaluating physiological signal salience for estimating metabolic energy cost from wearable sensors.*

After preprocessing, each subject’s data should be saved as a **CSV file** with the following structure:

| Column | Description | Type |
|--------|--------------|------|
| Time (s) | Time in seconds | float |
| Interpolated Values_Waist Acceleration | Waist acceleration | float |
| Interpolated Values_Chest Acceleration | Chest acceleration | float |
| Interpolated Values_Left Ankle Acceleration | Left ankle acceleration | float |
| Interpolated Values_right Ankle Acceleration | Right ankle acceleration | float |
| Interpolated_Left Leg (1–7) | Left leg kinematic features | float |
| Interpolated_Right Leg (1–7) | Right leg kinematic features | float |
| Interpolated_gaussian_EMG_magnitude_left/right | EMG signal magnitudes | float |
| Interpolated Values_left/right wrist Acceleration | Wrist accelerations | float |
| Interpolated Values_left/right wrist electrodermal | EDA (electrodermal activity) | float |
| Interpolated Values_left/right wrist Temperature | Skin temperature | float |
| Interpolated Values_VO2 | Oxygen consumption | float |
| Interpolated Values_Breath Frequency | Breathing frequency | float |
| Interpolated Values_Minute Ventilation | Minute ventilation | float |
| Interpolated Values_SpO2 | Blood oxygen saturation | float |
| Interpolated Values_Heart Rate | Heart rate | float |
| Interpolated Values_Energy expenditures | Energy expenditure | float |
| Activity Code | Encoded activity type | int |

Each preprocessed CSV file will contain **36 columns** and approximately **6488 rows** (per subject).

## 📘 Important Note on Signal Names

Make sure that the **names of the signals** in your CSV files are **identical** to the signal names used in the code.  
You can verify the correct names in the Jupyter notebook file in each model folder.

This consistency is critical for correct data loading and model execution.


