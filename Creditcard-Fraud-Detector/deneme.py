import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings
warnings.filterwarnings('ignore')


filename = "creditcard_fraud.csv"
df = pd.read_csv(filename)
print(df.head())

#-------------------------------------------------

# Count the number of fraudulent and non-fraudulent transactions
counts = df['Class'].value_counts()

# Get the labels and sizes for the pie chart
labels = counts.index
sizes = counts.values

# Plot the pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Ratio of Fraudalent to Non-Fraudalent Transactions')
plt.show()

#-----------------------------------------------------

from sklearn.model_selection import train_test_split
# Drop the Time column
df = df.drop(columns=['Time'])

# Split the data into train, validation, and test sets
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)

#--------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Build the Random Forest model
rf_model = RandomForestClassifier(random_state=0)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Build the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=0)

# Fit the model on the training data
gb_model.fit(X_train, y_train)

#---------------------------------------------------
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
# Make predictions on the validation data using the Random Forest model
y_pred_rf = rf_model.predict(X_val)

# Make predictions on the validation data using the Gradient Boosting model
y_pred_gb = gb_model.predict(X_val)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the confusion matrix for the Random Forest model
plot_confusion_matrix(rf_model, X_val, y_val, ax=ax1)
ax1.set_title('Random Forest')

# Plot the confusion matrix for the Gradient Boosting model
plot_confusion_matrix(gb_model, X_val, y_val, ax=ax2)
ax2.set_title('Gradient Boosting')

plt.show()
#-----------------------------------------------------------
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc, precision_recall_curve

# Calculate the false positive rate, true positive rate, and thresholds for the Random Forest model
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_val, rf_model.predict_proba(X_val)[:, 1])

# Calculate the AUC for the Random Forest model
auc_rf = auc(fpr_rf, tpr_rf)

# Calculate the false positive rate, true positive rate, and thresholds for the Gradient Boosting model
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_val, gb_model.predict_proba(X_val)[:, 1])

# Calculate the AUC for the Gradient Boosting model
auc_gb = auc(fpr_gb, tpr_gb)

# Plot the ROC curves
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.3f})'.format(auc_rf))
plt.plot(fpr_gb, tpr_gb, label='Gradient Boosting (AUC = {:.3f})'.format(auc_gb))

# Add the legend and diagonal line
plt.legend()
plt.plot([0, 1], [0, 1], 'k--')

# Set the axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Random Forest and Gradient Boosting Models')

plt.show()
#-------------------------------------------------------------------------
# Import the GridSearchCV function
from sklearn.model_selection import GridSearchCV

# Set up the parameter grid for the random forest classifier
param_grid_rf = {'n_estimators': [50, 100, 300, 500],
                 'max_features': [5, 7, 10, 25]}

# Create the random forest classifier
rf_classifier = RandomForestClassifier(random_state=0)

# Create the grid search object
grid_search_rf = GridSearchCV(estimator=rf_classifier,
                              param_grid=param_grid_rf,
                              scoring='average_precision',
                              cv=5,
                              n_jobs=-1)

# Fit the grid search object to the training data
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and the best score
print('Best Parameters for Random Forest:', grid_search_rf.best_params_)
print('Best AUPRC for Random Forest:', grid_search_rf.best_score_)
#-------------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter ranges for the Random Forest model
param_grid_rf = {
    'n_estimators': [50, 100, 300, 500],
    'max_features': [5, 7, 10, 25]
}

# Create the grid search object for the Random Forest model
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid_rf, scoring='average_precision', cv=5, return_train_score=True)

# Fit the grid search object to the training data
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters and best AUPRC score for the Random Forest model
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

# Define the hyperparameter ranges for the Gradient Boosting model
param_grid_gb = {
    'n_estimators': [50, 100, 300, 500],
    'max_features': [5, 7, 10, 25]
}

# Create the grid search object for the Gradient Boosting model
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid_gb, scoring='average_precision', cv=5, return_train_score=True)

# Fit the grid search object to the training data
grid_search_gb.fit(X_train, y_train)

# Get the best hyperparameters and best AUPRC score for the Gradient Boosting model
best_params_gb = grid_search_gb.best_params_
best_score_gb = grid_search_gb.best_score_

# Extract the best parameters and scores from the grid search object
best_params_rf = grid_search_rf.best_params_
best_scores_rf = grid_search_rf.best_score_

# Extract the list of n_estimators values
n_estimators_values_rf = param_grid_rf['n_estimators']

# Extract the list of max_features values
max_features_values_rf = param_grid_rf['max_features']

# Set up the subplot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the n_estimators vs AUPRC scores for the random forest classifier
ax1.plot(n_estimators_values_rf, best_scores_rf, 'o-')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('AUPRC')
ax1.set_title('Random Forest')

# Plot the max_features vs AUPRC scores for the random forest classifier
ax2.plot(max_features_values_rf, best_scores_rf, 'o-')
ax2.set_xlabel('max_features')
ax2.set_ylabel('AUPRC')
ax2.set_title('Random Forest')

# Set the y-axis limits
ax1.set_ylim([0.7, 1.0])
ax2.set_ylim([0.7, 1.0])

# Set the y-axis tick values
ax1.set_yticks(np.around(np.linspace(0.7, 1.0, 6), decimals=2))
ax2.set_yticks(np.around(np.linspace(0.7, 1.0, 6), decimals=2))

# Show the plot
plt.show()

#-------------------------------------------------------------------------

# Get the best model from the grid search object
best_model = grid_search_rf.best_estimator_

# Combine the train and validation data
X_train_val = np.concatenate((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))

# Retrain the best model on the combined train + validation data
best_model.fit(X_train_val, y_train_val)

#-------------------------------------------------------------------------

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Plot the confusion matrix
plot_confusion_matrix(best_model, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

#-------------------------------------------------------------------------

# Make predictions on the test data
y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]

# Plot the ROC curve
roc_curve_display = RocCurveDisplay(y_test, y_score, pos_label=1)
roc_curve_display.plot()

# Plot the precision-recall curve
precision_recall_display = PrecisionRecallDisplay(y_test, y_score, pos_label=1)
precision_recall_display.plot()

# Show the plots
plt.show()