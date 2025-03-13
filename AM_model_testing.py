import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.inspection import PartialDependenceDisplay

# Load CSV file
df = pd.read_csv("cleaned_data.csv")


# Change desired features as needed
feature_categories = ["Administrative", "Administrative_Duration", "Informational",
                      "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                      "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", 
                      "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
                      "Weekend"]

target = "Revenue"

# Encode categorical features
le_dict = {}
for col in feature_categories:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le


# Define features (X) and our target (Y)
X = df.drop(columns=[target])
Y = df[target]

# Split data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# ---CASE 1: Baseline Model---
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, Y_train)
Y_pred_default = rf_default.predict(X_test)
print("Baseline Model Accuracy: ", accuracy_score(Y_test, Y_pred_default))
print("Baseline ROC AUC: ", roc_auc_score(Y_test, rf_default.predict_proba(X_test)[:, 1]))
print("\nClassification Report:\n ", classification_report(Y_test, Y_pred_default))

# Baseline Visualization
display = ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_default)
plt.title("Baseline Model Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(rf_default, X_test, Y_test)

# ---CASE 2: Testing different numbers of trees---
n_estimators_list = [50, 100, 200, 500] # Adjust n as needed
roc_auc_scores = []

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    roc_auc = roc_auc_score(Y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc_scores.append(roc_auc)
    print(f"n_estimators={n} --> Accuracy: {accuracy_score(Y_test, Y_pred):.4f}, ROC AUC: {roc_auc:.4f}")
    
# n Trees visualization
plt.figure(figsize=(8, 6))
plt.plot(n_estimators_list, roc_auc_scores)
plt.xlabel("n")
plt.ylabel("ROC AUC Score")
plt.title("Effect of n_estimators on ROC AUC")
plt.show()

# ---CASE 3: Limiting depth of trees---
max_depth_list = [None, 10, 20, 30] # Adjust depth as needed
roc_auc_scores_depth = []

for md in max_depth_list:
    rf = RandomForestClassifier(max_depth=md, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    roc_auc = roc_auc_score(Y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc_scores_depth.append(roc_auc)
    print(f"max_depth={md} --> Accuracy: {accuracy_score(Y_test, Y_pred):.4f}, ROC AUC: {roc_auc:.4f}")
    
labels = ['0' if md is None else str(md) for md in max_depth_list] # List comprehension for setting a 'None' value to a label of 0

# Visualization
plt.figure(figsize=(8, 6))
plt.plot(labels, roc_auc_scores_depth, marker='o')
plt.xlabel("max_depth")
plt.ylabel("ROC AUC Score")
plt.title("Effect of max_depth on ROC AUC")
plt.show()
    
# ---CASE 4: Adjust min number of samples to split node---
min_samples_split_list = [2, 5, 10] # Adjust number of samples as needed
roc_auc_scores_split = []

for mss in min_samples_split_list:
    rf = RandomForestClassifier(min_samples_split=mss, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    print(f"min_samples_split={mss} --> Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
    auc_score = roc_auc_score(Y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc_scores_split.append(auc_score)
    
# Visualization    
plt.figure(figsize=(8,6))
plt.plot(min_samples_split_list, roc_auc_scores_split, marker='o')
plt.xlabel("min_samples_split")
plt.ylabel("ROC AUC Score")
plt.title("Effect of min_samples_split on ROC AUC")
plt.show()
    
# ---CASE 5: Testing different values for minimum number of samples required to be at a leaf node---
min_samples_leaf_list = [1, 2, 4]   # Adjust min number of samples as needed
roc_auc_scores_leaf = []

for msl in min_samples_leaf_list:
    rf = RandomForestClassifier(min_samples_leaf=msl, random_state=42)
    rf.fit(X_train, Y_train)
    auc_score = roc_auc_score(Y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc_scores_leaf.append(auc_score)
    Y_pred = rf.predict(X_test)
    print(f"min_samples_leaf={msl} --> Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

# Visualization    
plt.figure(figsize=(8, 6))
plt.plot(min_samples_leaf_list, roc_auc_scores_leaf, marker='o')
plt.xlabel("min_samples_leaf")
plt.ylabel("ROC AUC Score")
plt.title("Effect of min_samples_leaf on ROC AUC")
plt.show()
    
# ---CASE 6: Balance class weights---
rf_non_balanced = RandomForestClassifier(random_state=42)
rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_non_balanced.fit(X_train, Y_train)
rf_balanced.fit(X_train, Y_train)
Y_pred_non_balanced = rf_non_balanced.predict(X_test)
Y_pred_balanced = rf_balanced.predict(X_test)
print("Balanced Class Weight --> Accuracy: ", accuracy_score(Y_test, Y_pred_balanced))
print("Balanced Class Weight --> ROC AUC: ", roc_auc_score(Y_test, rf_balanced.predict_proba(X_test)[:,1]))

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_non_balanced, ax=ax[0])
ax[0].set_title("Non-balanced Class Weights")
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_balanced, ax=ax[1])
ax[1].set_title("Balanced Class Weights")
plt.show()

# ---CASE 7: Select subset of features that contribute the most to model---
rf_for_rfe = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rf_for_rfe, n_features_to_select=10)    # Can change n_features_to_select accordingly
rfe.fit(X_train, Y_train)
selected_features = X_train.columns[rfe.support_].tolist()
print("Selected Features: ", selected_features)

# Train new Random Forest on newly selected features
rf_rfe = RandomForestClassifier(random_state=42)
rf_rfe.fit(X_train[selected_features], Y_train)
Y_pred_rfe = rf_rfe.predict(X_test[selected_features])
print("RFE Model Accuracy: ", accuracy_score(Y_test, Y_pred_rfe))

importances = rf_rfe.feature_importances_
importance_df = pd.DataFrame({'Feature': selected_features, 'Significance': importances}).sort_values(by='Significance', ascending=True)

# Visualization
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Feature Significance")
plt.title("RFE Selected Features")
plt.show()

# Partial Dependence Plot for a provided feature
selected_feature = 'ProductRelated' # Replace 'ProductRelated' with any feature name provided by the data
PartialDependenceDisplay.from_estimator(rf_default, X_train, [selected_feature])
plt.title(f"Partial Dependence of feature {selected_feature}")
plt.show()