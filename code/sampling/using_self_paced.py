"""
NOTE:
The implementation of Self Paced Ensemble has been used from the Github 
repository: https://github.com/ZhiningLiu1998/self-paced-ensemble
"""
import warnings

import pandas as pd
import numpy as np
from sampling.self_paced_ensemble import self_paced_ensemble
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# NOTE: Updated the BASE_PATH as per your system before execution.
base_path = "/Users/Pratik/Data"
data_path = base_path + "/banksim1/bs140513_032310.csv"
raw_data = pd.read_csv(data_path)


# from `bank_sim_dat_exp.ipynb` previous analysis
def cat_amount(v, mean, median):
    res = ""
    if v > mean:
        res = "above_mean"
    elif v < median:
        res = "below_median"
    elif median <= v <= mean:
        res = "in_between"
    return res


amount_data = raw_data["amount"]
mean_amount = amount_data.mean()
median_amount = amount_data.median()
raw_data["amount_cat"] = np.vectorize(cat_amount) \
    (raw_data["amount"].values, mean_amount, median_amount)

pre_data = raw_data[["step", "customer", "age", "gender", "merchant", "category", "amount_cat", "fraud"]]
fraud_data = pre_data[pre_data["fraud"] == 1]
non_fraud_data = pre_data[pre_data["fraud"] == 0]

feat_cols = fraud_data.columns
print("List of feature columns used: {}".format(feat_cols))

f_train, f_test = train_test_split(fraud_data, test_size=0.2)
nf_train, nf_test = train_test_split(non_fraud_data, test_size=0.2)

train_df = pd.concat([f_train, nf_train]).sample(frac=1)
# Accuracy score for only the fraud rows is asked to be presented.
test_df = pd.concat([f_train, f_test]).sample(frac=1)

X_train = train_df.iloc[:, :-1].values
y_train = train_df["fraud"].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df["fraud"].values

# Transform categorical columns for training data
feat_cols = ["step", "customer", "age", "merchant", "category", "amount_cat"]
label_ens = []
for i in range(0, len(feat_cols)):
    en = LabelEncoder()
    X_train[:, i] = en.fit_transform(X_train[:, i])
    label_ens.insert(i, en)

one_hot_en = OneHotEncoder(handle_unknown='ignore')
X_train = one_hot_en.fit_transform(X_train)

# Transform categorical columns for test data
for i in range(0, len(feat_cols)):
    X_test[:, i] = label_ens[i].transform(X_test[:, i])

X_test = one_hot_en.transform(X_test)

# Testing Self Paced Ensemble Method
spe = self_paced_ensemble.SelfPacedEnsemble().fit(X_train, y_train)
y_pred = spe.predict(X_test)

print("\nModel trained using Self Paced Ensemble technique:")
recall = recall_score(y_test, y_pred, average="binary")
precision = precision_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Recall score for the base model is {0}".format(round(recall, 4)))
print("Precision score for the base model is {0}".format(round(precision, 4)))
print("F1 score for the base model is {0}".format(round(f1, 4)))
print("TP: {0}\tFP: {1}\nFN: {2}\tTN: {3}".format(tp, fp, fn, tn))
