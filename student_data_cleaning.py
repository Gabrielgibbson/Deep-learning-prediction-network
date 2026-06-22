import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("StudentPerformanceFactors.csv")
# print(df.describe())

df = df.dropna()
# print(df.isna().sum())
# print(df.info())

columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Parental_Education_Level', 'Distance_from_Home']
encoders = {}
for column in columns:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  encoders[column] = le
corr = df.corr(numeric_only=True)
# print(corr['Exam_Score'].sort_values(ascending=True))
# print(df['Exam_Score'].corr(df))
df.to_csv("clean_student_data.csv", index=False)