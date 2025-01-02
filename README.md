```python
# Hybrid Polynomial Ensemble Model (HPEM)

This repository contains the implementation of the Hybrid Polynomial Ensemble Model (HPEM) for cardiovascular disease prediction. 
The model combines multiple classifiers including Random Forest, Gradient Boosting, and Support Vector Machines (SVM) with a meta-learner (XGBoost).

## Features
- Advanced feature engineering and selection.
- Integration of multiple datasets.
- Comprehensive model evaluation using metrics like accuracy, precision, recall, and F1-Score.

## Prerequisites
1. Install Python 3.8+
2. Clone the repository:
   ```
   git clone <repository_url>
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Place datasets (`heart.csv`, `cardio.csv`, `alt_heart.csv`) in the `datasets/` folder.

## Run the Model
Run the following command to train and evaluate the model:
```
python HPEM-Model-Code.py
```

## Folder Structure
- `datasets/`: Contains all input datasets (`heart.csv`, `cardio.csv`, `alt_heart.csv`).
- `hpem_model.py`: Main script to run the HPEM model.
- `notebooks/`: Contains Jupyter Notebooks for exploratory data analysis (EDA).
- `requirements.txt`: Python dependencies for the project.

## License
This project is licensed under the MIT License.

```
