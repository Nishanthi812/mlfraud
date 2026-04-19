import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

data = pd.read_csv("datasetm.csv")

print("Dataset Preview:")
print(data.head())

X = data.drop("risk_label", axis=1)
y = data["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=3)
svm = SVC(kernel='poly', degree=2, C=0.8, probability=True)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

lr_rec = recall_score(y_test, lr_pred)
rf_rec = recall_score(y_test, rf_pred)
svm_rec = recall_score(y_test, svm_pred)

lr_f1 = f1_score(y_test, lr_pred)
rf_f1 = f1_score(y_test, rf_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("\n=== Model Performance ===")
print("Logistic Regression -> Accuracy:", round(lr_acc,2), " Recall:", round(lr_rec,2), " F1:", round(lr_f1,2))
print("Random Forest -> Accuracy:", round(rf_acc,2), " Recall:", round(rf_rec,2), " F1:", round(rf_f1,2))
print("SVM -> Accuracy:", round(svm_acc,2), " Recall:", round(svm_rec,2), " F1:", round(svm_f1,2))

print("\n=== Enter Transaction Details ===")

amount = float(input("Transaction Amount: "))
fee = float(input("Transaction Fee: "))
input_count = int(input("Input Count: "))
output_count = int(input("Output Count: "))
pattern = float(input("Unusual Pattern Score (0-1): "))
velocity = float(input("Transaction Velocity (24h): "))

user_data = pd.DataFrame([[amount, fee, input_count, output_count, pattern, velocity]],
                         columns=X.columns)

user_data = scaler.transform(user_data)

prediction = rf.predict(user_data)[0]

print("\n=== Prediction Result ===")

if prediction == 1:
    print("Fraudulent Transaction Detected")
else:
    print("Legitimate Transaction")

models = ["LR", "RF", "SVM"]
accuracies = [lr_acc, rf_acc, svm_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()

feature_importance = pd.Series(rf.feature_importances_, index=data.drop("risk_label", axis=1).columns)
feature_importance.sort_values(ascending=False).plot(kind='bar')

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

cm_lr = confusion_matrix(y_test, lr_pred)
ConfusionMatrixDisplay(cm_lr).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

cm_rf = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(cm_rf).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

cm_svm = confusion_matrix(y_test, svm_pred)
ConfusionMatrixDisplay(cm_svm).plot()
plt.title("SVM Confusion Matrix")
plt.show()