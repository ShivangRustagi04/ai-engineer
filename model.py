import pandas as pd
import json
import os

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def run_kmeans_clustering(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model

def run_decision_tree_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy
from jinja2 import Template
import pandas as pd

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

from jinja2 import Template

def generate_report(summary, df, visualizations=['static/pairplot.png']):
    data_summary = generate_data_summary(df)

    template = Template("""
    <html>
        <head><title>Analysis Report</title></head>
        <body>
            <h1>Data Analysis Report</h1>
            <h2>Dataset Summary</h2>
            <pre>{{ data_summary }}</pre>
            <h2>Analysis Summary</h2>
            <p>{{ summary }}</p>
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <img src="/static/pairplot.png" style="width:100%;max-width:600px;">
            {% endfor %}
        </body>
    </html>
    """)

    report = template.render(data_summary=data_summary, summary=summary, visualizations=visualizations)
    with open('report.html', 'w') as f:
        f.write(report)

    print("Report generated and saved as 'report.html'")



from sklearn.preprocessing import LabelEncoder
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(df):
    output_path = os.path.join('static', 'pairplot.png')
    sns.pairplot(df)
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved at '{output_path}'")




def main():
    if 'ipykernel' in sys.modules:
        args = argparse.Namespace(
            file='olympics2024.csv',
            analyze='clustering',
            target='Gold'
        )
    else:
        parser = argparse.ArgumentParser(description="AI Employee - Data Analysis")
        parser.add_argument('--file', type=str, help="Path to the data file")
        parser.add_argument('--analyze', type=str, choices=['regression', 'clustering', 'classification'],
                            help="Type of analysis to run")
        parser.add_argument('--target', type=str, help="Target column for analysis")
        args = parser.parse_args()
    df = load_data(args.file)
    df = clean_data(df)

    print("Available columns in the dataset:", df.columns.tolist())

    if args.target not in df.columns:
        raise KeyError(f"Target column '{args.target}' not found in the dataset.")

    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != args.target: 
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    if args.analyze == 'regression':
        X = df.drop(columns=[args.target])
        y = df[args.target]
        model = run_linear_regression(X, y)
        print("Linear Regression model coefficients:", model.coef_)

    elif args.analyze == 'clustering':
        X = df
        model = run_kmeans_clustering(X)
        print("K-Means Cluster Centers:", model.cluster_centers_)

    elif args.analyze == 'classification':
        X = df.drop(columns=[args.target])
        y = df[args.target]
        model, accuracy = run_decision_tree_classification(X, y)
        print(f"Decision Tree Classification Accuracy: {accuracy}")

    summary = "Analysis summary goes here."
    generate_visualizations(df)
    generate_report(summary,df)

if __name__ == '__main__':
    main()
import unittest

class TestAIModule(unittest.TestCase):

    def test_load_data_csv(self):
        df = load_data('olympics2024.csv')  
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        df = pd.DataFrame({'A': [1, 2, None], 'B': [None, 2, 3]})
        clean_df = clean_data(df)
        self.assertFalse(clean_df.isnull().values.any())

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIModule)
    unittest.TextTestRunner().run(suite)

run_tests()