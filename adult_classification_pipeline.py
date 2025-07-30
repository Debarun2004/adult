import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Step 1: Load and inspect data
data = pd.read_csv("adult.csv")

# Step 2: Clean and preprocess
data['occupation'].replace({' ?': 'Others'}, inplace=True)
data['workclass'].replace({' ?': 'Notlisted'}, inplace=True)
data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
data.drop(columns=['education'], inplace=True)
data = data[(data['age'] >= 17) & (data['age'] <= 75)]

# Step 3: Encode categorical columns
label_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# Step 4: Feature-target split and scaling
x = data.drop(columns=['income'])
y = data['income']
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Step 5: Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=23, stratify=y)

# Step 6: Model training and evaluation
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "MLP Classifier": MLPClassifier(solver='adam', hidden_layer_sizes=(5, 2), random_state=2, max_iter=2000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
reports = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    reports[name] = classification_report(y_test, preds, output_dict=True)
    conf_matrices[name] = confusion_matrix(y_test, preds)

# Step 7: Accuracy comparison plot
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy Score")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Show best model report and confusion matrix
best_model = max(results, key=results.get)
best_cm = conf_matrices[best_model]
print(f"\nBest Model: {best_model}\n")
print("Classification Report:\n")
print(pd.DataFrame(reports[best_model]).transpose())

# Confusion Matrix Heatmap
sns.heatmap(best_cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Confusion Matrix for {best_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
