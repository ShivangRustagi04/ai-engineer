# Flask Data Analysis Application

This Flask application allows users to upload datasets and perform various data analysis tasks such as linear regression, K-means clustering, and decision tree classification. The application generates summaries, visualizations, and a downloadable HTML report based on the selected analysis.

## Features

- **File Upload**: Upload your dataset (CSV format).
- **Data Analysis**:
  - **Linear Regression**
  - **K-Means Clustering**
  - **Decision Tree Classification**
- **Automatic Report Generation**: Generates an HTML report with a summary and visualizations.
- **Downloadable Report**: Download the generated report in HTML format.

## Requirements

- Python 3.7+
- Flask
- pandas
- scikit-learn
- jinja2

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ShivangRustagi04/ai-engineer.git
   ```


```bash
   .
├── app.py                # Main application script
├── model.py              # Module containing data processing and analysis functions
├── uploads/              # Directory where uploaded files are saved
├── static/               # Directory for static files (e.g., visualizations)
├── templates/            # Directory for HTML templates
└── report.html           # Generated report file
```

2. Install Libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. To run code
   ```bash
   python app.py
   ```
   

   
