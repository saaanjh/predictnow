from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

df = pd.read_csv('dataset.csv')

label_encoders = {}
for column in ['CATEGORY', 'SEASON', 'FIT', 'PATTERN', 'FABRIC COMPOSITION', 'Denim Wash', 'COLOR']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le


features = df[['CATEGORY', 'MRP', 'SEASON']]


target = df['FIT']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


fit_model = RandomForestClassifier(random_state=42)
fit_model.fit(X_train, y_train)


predictions = fit_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy for FIT model: {accuracy}')


target = df['PATTERN']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


pattern_model = RandomForestClassifier(random_state=42)
pattern_model.fit(X_train, y_train)


predictions = pattern_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy for FIT model: {accuracy}')


target_fabric = df['FABRIC COMPOSITION']


X_train_fabric, X_test_fabric, y_train_fabric, y_test_fabric = train_test_split(features, target_fabric, test_size=0.3, random_state=42)


model_fabric = RandomForestClassifier(random_state=42)
model_fabric.fit(X_train_fabric, y_train_fabric)


predictions_fabric = model_fabric.predict(X_test_fabric)
accuracy_fabric = accuracy_score(y_test_fabric, predictions_fabric)
print(f'Accuracy for FABRIC COMPOSITION model: {accuracy_fabric}')


target_color = df['COLOR']


X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(features, target_color, test_size=0.3, random_state=42)


model_color = RandomForestClassifier(random_state=42)
model_color.fit(X_train_color, y_train_color)


predictions_color = model_color.predict(X_test_color)
accuracy_color = accuracy_score(y_test_color, predictions_color)
print(f'Accuracy for COLOR model: {accuracy_color}')



target_color = df['Denim Wash']


X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(features, target_color, test_size=0.3, random_state=42)


model_denim = RandomForestClassifier(random_state=42)
model_denim.fit(X_train_color, y_train_color)


predictions_color = model_denim.predict(X_test_color)
accuracy_color = accuracy_score(y_test_color, predictions_color)
print(f'Accuracy for COLOR model: {accuracy_color}')


def predict_attributes(category, mrp, season):
    
    encoded_inputs = []
    for encoder, input_value in zip(['CATEGORY', 'MRP', 'SEASON'], [category, mrp, season]):
        if encoder == 'MRP':  
            encoded_inputs.append(input_value)
        else:
            encoded_input = label_encoders[encoder].transform([input_value])[0]
            encoded_inputs.append(encoded_input)

    
    encoded_inputs = [encoded_inputs] 

    
    fit_pred = fit_model.predict(encoded_inputs)[0]
    fit_pred = label_encoders['FIT'].inverse_transform([fit_pred])[0]

    
    pattern_pred = pattern_model.predict(encoded_inputs)[0]
    pattern_pred = label_encoders['PATTERN'].inverse_transform([pattern_pred])[0]

    
    fabric_comp_pred = model_fabric.predict(encoded_inputs)[0]
    fabric_comp_pred = label_encoders['FABRIC COMPOSITION'].inverse_transform([fabric_comp_pred])[0]

    
    denim_wash_pred = model_denim.predict(encoded_inputs)[0]  
    denim_wash_pred = label_encoders['Denim Wash'].inverse_transform([denim_wash_pred])[0]

    
    color_pred = model_color.predict(encoded_inputs)[0]  
    color_pred = label_encoders['COLOR'].inverse_transform([color_pred])[0]

    return {
        'FIT': fit_pred,
        'PATTERN': pattern_pred,
        'FABRIC COMPOSITION': fabric_comp_pred,
        'DENIM WASH': denim_wash_pred,
        'COLOR': color_pred
    }

def predict_top_n_attributes(category, mrp, season, n=2):
    # Encode inputs
    encoded_inputs = []
    for encoder, input_value in zip(['CATEGORY', 'MRP', 'SEASON'], [category, mrp, season]):
        if encoder == 'MRP':  
            encoded_inputs.append(input_value)
        else:
            encoded_input = label_encoders[encoder].transform([input_value])[0]
            encoded_inputs.append(encoded_input)

    
    encoded_inputs = [encoded_inputs]  

    predictions = {}

    
    fit_probs = fit_model.predict_proba(encoded_inputs)[0]
    top_n_fit_indices = fit_probs.argsort()[-n:][::-1]  
    top_n_fit_preds = label_encoders['FIT'].inverse_transform(top_n_fit_indices)
    predictions['FIT'] = top_n_fit_preds.tolist()

    
    pattern_probs = pattern_model.predict_proba(encoded_inputs)[0]
    top_n_pattern_indices = pattern_probs.argsort()[-n:][::-1]
    top_n_pattern_preds = label_encoders['PATTERN'].inverse_transform(top_n_pattern_indices)
    predictions['PATTERN'] = top_n_pattern_preds.tolist()

    
    fabric_comp_probs = model_fabric.predict_proba(encoded_inputs)[0]
    top_n_fabric_comp_indices = fabric_comp_probs.argsort()[-n:][::-1]
    top_n_fabric_comp_preds = label_encoders['FABRIC COMPOSITION'].inverse_transform(top_n_fabric_comp_indices)
    predictions['FABRIC COMPOSITION'] = top_n_fabric_comp_preds.tolist()

    
    denim_wash_probs = model_denim.predict_proba(encoded_inputs)[0]
    top_n_denim_wash_indices = denim_wash_probs.argsort()[-n:][::-1]
    top_n_denim_wash_preds = label_encoders['Denim Wash'].inverse_transform(top_n_denim_wash_indices)
    predictions['DENIM WASH'] = top_n_denim_wash_preds.tolist()

   
    color_probs = model_color.predict_proba(encoded_inputs)[0]
    top_n_color_indices = color_probs.argsort()[-n:][::-1]
    top_n_color_preds = label_encoders['COLOR'].inverse_transform(top_n_color_indices)
    predictions['COLOR'] = top_n_color_preds.tolist()

    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    import pandas as pd
    df2 = pd.read_csv('dataset.csv')
    
    unique_categories = sorted(df2['CATEGORY'].unique().tolist())
    unique_mrps = sorted(df2['MRP'].unique().astype(str).tolist())  
    unique_seasons = sorted(df2['SEASON'].unique().tolist())

    print(unique_categories)

    if request.method == 'POST':
        category = request.form['category']
        mrp = request.form['mrp']
        season = request.form['season']
        predictions = predict_top_n_attributes(category, mrp, season, n=2)
        print(predictions)
        return render_template('index.html', predictions=predictions, unique_categories=unique_categories, unique_mrps=unique_mrps, unique_seasons=unique_seasons)
    return render_template('index.html', unique_categories=unique_categories, unique_mrps=unique_mrps, unique_seasons=unique_seasons)