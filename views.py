import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask import Blueprint, render_template, request
from flask_login import login_required, current_user


df_diabetes = pd.read_csv('diabetes_prediction_dataset.csv')


X_diabetes = df_diabetes[['bmi']]
y_diabetes = df_diabetes['diabetes']


preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  
])

preprocessor.fit(X_diabetes)


rf_model_diabetes = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_model_diabetes.fit(X_diabetes, y_diabetes)


views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
def home():
    return render_template("landingpage.html", user=current_user)

@views.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

@views.route('/healthtracking', methods=['GET', 'POST'])
@login_required
def healthtracking():
    return render_template("healthtracking.html", user=current_user)   

@views.route('/healthdata', methods=['GET', 'POST'])
@login_required
def healthdata():
    bmi = float(request.form['bmi'])
    
  
    input_data = pd.DataFrame({'bmi': [bmi]})
    
    input_data_transformed = preprocessor.transform(input_data)
    prediction = rf_model_diabetes.predict(input_data_transformed)
    priority = 1 if 1 in prediction else 0
    print("priority is :",priority)
    if(priority==1):
      return render_template('result.html')
    else:
       return render_template('result0.html')  
