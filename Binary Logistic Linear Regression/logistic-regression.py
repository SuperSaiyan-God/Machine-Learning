import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import logit
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import f1_score


diabetes = pd.read_csv("diabetes.csv")

diabetes["diabetes"] = diabetes["diabetes"].map({"neg":0, "pos":1})

train_data, test_data = train_test_split(diabetes,
                                         test_size = 0.20,
                                         random_state = 42)


# Fitting Logistic Regression
formula = ('diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age')

model = logit(formula = formula, data = train_data)
model.fit()

# Compute prediction
prediction = model.predict(exog = test_data)

# Define the cutoff
cutoff = 0.5

# Compute class predictions: y_prediction
y_prediction = np.where(prediction > cutoff, 1, 0)

# Assign actual class labels from the test sample to y_actual
y_actual = test_data["diabetes"]


# Compute and print confusion matrix using crosstab function
conf_matrix = pd.crosstab(y_actual, y_prediction,
                       rownames = ["Actual"], 
                       colnames = ["Predicted"], 
                       margins = True)
                      
# Print the confusion matrix
print(conf_matrix)

accuracy = accuracy_score(y_actual, y_prediction)

print('Accuracy: %.2f' % accuracy + "%")

print("F1 Score: {}".format(f1_score(y_actual, y_prediction)))

roc_auc = roc_auc_score(y_actual, y_prediction)
print('AUC: %.2f' % roc_auc + "%")

# Visualisation with plot_metric
bc = BinaryClassification(y_actual,
                          y_prediction,
                          labels = ["Class 1", "Class 2"])

# Figures
plt.figure(figsize = (5,5))
bc.plot_roc_curve()
plt.show()