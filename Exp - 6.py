import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the Iris dataset
dataset = pd.read_csv("/content/IRIS.csv")

# Split the dataset into features (X) and target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# K-Nearest Neighbors (KNN)
classifier_knn = KNeighborsClassifier(n_neighbors=5)
classifier_knn.fit(X_train, y_train)
accuracy_knn = classifier_knn.score(X_test, y_test)
print("Accuracy using KNN:", accuracy_knn)

# Naive Bayes
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
accuracy_nb = classifier_nb.score(X_test, y_test)
print("Accuracy using Naive Bayes:", accuracy_nb)

# Logistic Regression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)
accuracy_lr = classifier_lr.score(X_test, y_test)
print("Accuracy using Logistic Regression:", accuracy_lr)

# Decision Tree
classifier_dt = DecisionTreeClassifier(random_state=42)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy using Decision Tree:", accuracy_dt)

# Support Vector Machine (SVM)
classifier_svm = SVC(kernel='linear', random_state=42)
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy using SVM:", accuracy_svm)

# Random Forest
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy using Random Forest:", accuracy_rf)
