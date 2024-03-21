# Bayesian Network using pgmpy
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Age': [30, 45, 60, 25, 50],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Family_History': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Cholesterol': ['High', 'Normal', 'High', 'Normal', 'High'],
    'Blood_Pressure': ['High', 'Normal', 'Normal', 'Normal', 'High'],
    'Smoking': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Physical_Activity': ['Low', 'High', 'Low', 'High', 'Low'],
    'Diet': ['Unhealthy', 'Healthy', 'Healthy', 'Unhealthy', 'Healthy'],
    'Heart_Disease': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

# Define Bayesian Network structure
model = BayesianModel([
    ('Age', 'Cholesterol'),
    ('Age', 'Blood_Pressure'),
    ('Gender', 'Cholesterol'),
    ('Smoking', 'Cholesterol'),
    ('Physical_Activity', 'Blood_Pressure'),
    ('Diet', 'Cholesterol'),
    ('Family_History', 'Heart_Disease'),
    ('Cholesterol', 'Heart_Disease'),
    ('Blood_Pressure', 'Heart_Disease')
])

# Train the Bayesian Network
model.fit(data)

# Perform Variable Elimination to query probabilities
inference = VariableElimination(model)
prob_heart_disease = inference.query(variables=['Heart_Disease'], evidence={
                                    'Age': 40, 'Gender': 'Male', 'Family_History': 'Yes'})

print(prob_heart_disease)

# Principal Component Analysis (PCA) using scikit-learn
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load sample iris dataset
iris = load_iris()
X = iris.data

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
