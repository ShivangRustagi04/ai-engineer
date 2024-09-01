from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from jinja2 import Template

# Import functions from the model module
from model import (
    load_data, clean_data, run_linear_regression, run_kmeans_clustering,
    run_decision_tree_classification, generate_visualizations, generate_report
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def generate_report(summary, df, visualizations=['static/pairplot.png']):
    # Print the summary to debug
    print(f"Summary Content: {summary}")

    data_summary = generate_data_summary(df)

    template = Template("""
    <html>
        <head><title>Analysis Report</title></head>
        <body>
            <h1>Data Analysis Report</h1>
            <h2>Analysis Report</h2>
            <pre>{{ summary }}</pre>
            <h2>Dataset Summary</h2>
            <pre>{{ data_summary }}</pre>
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <img src="{{ viz }}" style="width:100%;max-width:600px;">
            {% endfor %}
        </body>
    </html>
    """)

    report = template.render(data_summary=data_summary, summary=summary, visualizations=visualizations)
    with open('report.html', 'w') as f:
        f.write(report)

def generate_data_summary(df):
    summary = {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().sum(),
        'basic_stats': df.describe().to_dict()
    }

    stats_summary = "\n".join([f"{key}: {value}" for key, value in summary['basic_stats'].items()])

    summary_text = f"""
    Dataset Summary:
    ----------------
    - Number of rows: {summary['num_rows']}
    - Number of columns: {summary['num_columns']}
    - Columns: {', '.join(summary['columns'])}
    - Total missing values: {summary['missing_values']}

    Basic Statistics:
    -----------------
    {stats_summary}
    """

    return summary_text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the file to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load the dataset into a DataFrame
            df = pd.read_csv(filepath)

            # Generate data summary
            data_summary = generate_data_summary(df)

            # Get analysis type and target column from form
            analysis_type = request.form.get('analysis_type')
            target_column = request.form.get('target_column')

            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found in the dataset.", 400

            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != target_column:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])

            # Run the selected analysis
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

            # Generate visualizations and the report
            generate_visualizations(df)
            generate_report(summary, df)

            # Render the report template with the generated summary
            return render_template('report.html', summary=summary, data_summary=data_summary)

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Dataset</title>
    </head>
    <body>
        <h1>Upload a dataset for analysis</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            
            <label for="analysis_type">Choose analysis type:</label>
            <select name="analysis_type" id="analysis_type">
                <option value="regression">Linear Regression</option>
                <option value="clustering">K-Means Clustering</option>
                <option value="classification">Decision Tree Classification</option>
            </select>

            <label for="target_column">Target column (for regression/classification/clustering):</label>
            <input type="text" id="target_column" name="target_column">
            
            <input type="submit" value="Upload and Analyze">
        </form>
    </body>
    </html>
    '''

@app.route('/download', methods=['GET'])
def download_report():
    path = "report.html"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
