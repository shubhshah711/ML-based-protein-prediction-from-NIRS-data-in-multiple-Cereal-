from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, send_file
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# Load the trained model
with open('new.pkl', 'rb') as f:
    model = pickle.load(f)


# Define a function to make predictions
def predict_target(data):
    print("hi")
    print(data)
    df=data
    print(model)
    # Standardize the features
    y_test=pd.read_excel('y_test3.xlsx')


    predictions = model.predict(df)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, predictions)), '.3f'))
    print("\nRMSE: ", rmse)

    r2 = r2_score(y_test, predictions)
    print(f'R² Score: {r2}')
    print(predictions)
    return predictions

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)
            # Read the uploaded Excel file
            print(filename)
            df = pd.read_excel(filename)
            print(df)
            # Perform prediction
            predictions = predict_target(df)
            print(predictions)
            dataframe = pd.DataFrame({'Predicted Protein Content': predictions})
            y_test = pd.read_excel('y_test.xlsx')  # Load y_test from file, adjust filename as needed
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            # Save the DataFrame to an Excel file
            dataframe.to_excel('predicted_protein_content.xlsx', index=False)
            return render_template('result.html', rmse=rmse, r2=r2)

            # Display predictions
        return render_template('upload.html', message=" Please upload a file.")

@app.route('/download')
def download():
    return send_file('predicted_protein_content.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
