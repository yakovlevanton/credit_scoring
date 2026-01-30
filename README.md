# Credit Scoring
This project implements a binary classification task: predicting the probability  
of client default.

## Data
The dataset for this task can be downloaded from the following link:  
[Home Credit dataset](https://drive.google.com/drive/folders/17qNOwNyzjRDfdFF4EpZSR0JeSvl7Hj70?usp=sharing)  
It is recommended to unzip the archive into the `data/raw` directory in the root of the repository.  

The dataset consists of 8 `*.csv` files. The data schema is shown below:  
![data_scheme](images/data_scheme.png)

I used 4 of them: main application and client information in  
`application_train.csv` and `application_test.csv`, information about past  
debts to other credit organizations in `bureau.csv`, and information about  
previous applications to the same bank in `previous_application.csv`.

## Approach Used  
* **Data preprocessing**: removing features with >50% missing values, weak correlation with the target variable,
  strong correlation with each other, and low informativeness. Handling anomalies in the data, creating
  additional useful features and aggregated features based on other tables.  
* **Model**: `CatBoostClassifier`, since there are many categorical features and features with `NaN` values.  
* **Validation**: standard 80/20 split.  
* **Loss function**: `LogLoss` — a classic choice for binary classification.  
* **Evaluation metric**: `AUC ROC`, because it shows how well the model distinguishes between default and reliable  
  clients, regardless of the classification threshold (it is computed on probabilities, not labels), and is  
  robust to class imbalance.
  
## Results  
* Validation metric: `AUC ROC = 0.77`  
* Metric on the private test set (from Kaggle): `AUC ROC = 0.76`  
* Thus, the model demonstrated fairly good performance and generalization ability.  
* The contribution of the most important features to the model predictions is shown below:  
![shap_values](images/shap.png)

## How to Run  
1) Clone the repository  
   `git clone https://github.com/yakovlevanton/credit_scoring.git`  
   `cd credit_scoring`
2) Create and activate a virtual environment  
   `python -m venv .venv`  
   Linux/macOS:  
   `source .venv/bin/activate`  
   Windows:  
   `.venv\Scripts\activate`  
3) Install dependencies  
   `pip install -r requirements.txt`  
4) Download and unzip the archive from the link above  
5) Train the model:  
   `python run.py --mode train`  
6) Generate predictions:  
   `python run.py --mode predict --out <path/to/file/.csv>`  
   By default, they will be saved to `predictions/submission.csv`
