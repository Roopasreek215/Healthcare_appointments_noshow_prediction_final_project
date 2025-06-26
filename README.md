# Healthcare_appointments_noshow_prediction_final_project
Summary of your Healthcare No-Show Prediction project:

1. The project predicts if a patient will miss their medical appointment.
2. Data includes 110K+ records with features like age, gender, SMS reminders, and more.
3. Python was used for data preprocessing and modeling using Scikit-learn.
4. Missing values were handled, and datetime fields were engineered into "waiting days."
5. Categorical columns like Gender and Neighbourhood were encoded using one-hot encoding.
6. A Decision Tree Classifier was trained with `class_weight='balanced'` to handle imbalance.
7. The model achieved 86% recall for no-shows, successfully identifying high-risk patients.
8. Feature importance showed that SMS, age, and waiting time influenced no-shows most.
9. Power BI was used to visualize trends like no-shows by age, SMS received, and weekday.
10. The outcome supports better scheduling and reduces missed appointments.
