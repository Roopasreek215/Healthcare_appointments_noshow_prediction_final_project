
# Healthcare No-Show Prediction

# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# 2. Load dataset
df = pd.read_csv('appointments.csv')

# 3. Basic cleanup
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})  # Convert to 0/1

# Drop unnecessary ID columns
df = df.drop(['PatientId', 'AppointmentID'], axis=1)

# 4. Handle datetime columns
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Feature: Waiting days between scheduling and appointment
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Drop raw datetime columns (cannot be used directly in model)
df = df.drop(['ScheduledDay', 'AppointmentDay'], axis=1)

# 5. Encode categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood'], drop_first=True)

# 6. Split data
X = df.drop('No-show', axis=1)
y = df['No-show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
