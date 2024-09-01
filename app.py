from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from model import load_data, clean_data, run_linear_regression, run_kmeans_clustering, run_decision_tree_classification, generate_visualizations, generate_report
from sklearn.preprocessing import LabelEncoder
import sys
import argparse

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            return redirect(request.url)

        file = request.files['datafile']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            analysis_type = request.form.get('analysis_type')
            target_column = request.form.get('target_column')
            return redirect(url_for('analyze', filepath=filepath, analysis_type=analysis_type, target_column=target_column))

    return render_template('index.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    filepath = request.args.get('filepath')
    analysis_type = request.args.get('analysis_type')
    target_column = request.args.get('target_column')

    if not filepath or not os.path.exists(filepath):
        return "Error: File not found.", 400

    df = load_data(filepath)
    df = clean_data(df)

    if target_column not in df.columns:
        return f"Error: Target column '{target_column}' not found in the dataset.", 400

    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    if analysis_type == 'regression':
        X = df.drop(columns=[target_column])
        y = df[target_column]
        model = run_linear_regression(X, y)
        summary = f"Linear Regression model coefficients: {model.coef_}"

    elif analysis_type == 'clustering':
        X = df
        model = run_kmeans_clustering(X)
        summary = f"K-Means Cluster Centers: {model.cluster_centers_}"

    elif analysis_type == 'classification':
        X = df.drop(columns=[target_column])
        y = df[target_column]
        model, accuracy = run_decision_tree_classification(X, y)
        summary = f"Decision Tree Classification Accuracy: {accuracy}"

    generate_visualizations(df)
    generate_report(summary, df)
    
    return render_template('report.html', summary=summary)

@app.route('/download', methods=['GET'])
def download_report():
    path = "report.html"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
