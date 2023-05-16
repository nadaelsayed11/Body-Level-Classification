import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# read test data and perform inference
data=pd.read_csv('test.csv')
#data = data.sample(frac=1, random_state=42)
# drop Body_Level column if exists
if 'Body_Level' in data.columns:
    y=data['Body_Level']
    data=data.drop('Body_Level',axis=1)

# perform inference
loaded_model = joblib.load('lgbm_model.pkl')
# load encoder
le = joblib.load('label_encoder.joblib')

# convert Body Level to numeric values
data['Gender'] = pd.factorize(data['Gender'])[0]
#data['Body_Level'] = pd.factorize(data['Body_Level'])[0]
# H_Cal_Consump
data['H_Cal_Consump'] = pd.factorize(data['H_Cal_Consump'])[0]
# Alcohol_Consump
data['Alcohol_Consump'] = pd.factorize(data['Alcohol_Consump'])[0]
# Smoking
data['Smoking'] = pd.factorize(data['Smoking'])[0]
# Food_Between_Meals
data['Food_Between_Meals'] = pd.factorize(data['Food_Between_Meals'])[0]
# Fam_Hist
data['Fam_Hist'] = pd.factorize(data['Fam_Hist'])[0]
# H_Col_Burn
data['H_Cal_Burn'] = pd.factorize(data['H_Cal_Burn'])[0]
# Tranport
data['Transport'] = pd.factorize(data['Transport'])[0]

scaler = StandardScaler()
X = scaler.fit_transform(data)
# Use the loaded model to make predictions
y_pred = loaded_model.predict(X)

y_pred = le.inverse_transform(y_pred)
# save predictions in txt file
with open('predictions.txt', 'w') as f:
    for item in y_pred:
        f.write("%s\n" % item)    


