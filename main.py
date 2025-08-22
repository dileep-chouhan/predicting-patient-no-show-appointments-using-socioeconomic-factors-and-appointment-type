import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_patients = 500
data = {
    'Age': np.random.randint(18, 80, num_patients),
    'Income': np.random.randint(20000, 150000, num_patients),
    'TravelTime': np.random.randint(5, 120, num_patients),  # in minutes
    'AppointmentType': np.random.choice(['General Checkup', 'Specialist Visit', 'Surgery'], num_patients),
    'NoShow': np.random.choice([0, 1], num_patients, p=[0.8, 0.2]) # 80% show, 20% no-show
}
df = pd.DataFrame(data)
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['AppointmentType'], drop_first=True)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this synthetic data, no cleaning is explicitly needed.
# --- 3. Predictive Modeling ---
X = df.drop('NoShow', axis=1)
y = df['NoShow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
# Example: Visualize the relationship between income and no-show rate
plt.figure(figsize=(8, 6))
sns.boxplot(x='NoShow', y='Income', data=df)
plt.title('Income Distribution by No-Show Status')
plt.xlabel('No-Show (0=Show, 1=No-Show)')
plt.ylabel('Income')
plt.savefig('income_vs_noshow.png')
print("Plot saved to income_vs_noshow.png")
#Example: Visualize the relationship between travel time and no-show rate
plt.figure(figsize=(8,6))
sns.scatterplot(x='TravelTime', y='NoShow', data=df)
plt.title('Travel Time vs. No-Show Rate')
plt.xlabel('Travel Time (minutes)')
plt.ylabel('No-Show (0=Show, 1=No-Show)')
plt.savefig('traveltime_vs_noshow.png')
print("Plot saved to traveltime_vs_noshow.png")