# Simple Correlation Analysis Dashboard

A Streamlit-based dashboard for exploring correlations between variables in numeric datasets.

## Features

- **Interactive Correlation Heatmap**: Visualize correlation matrices with customizable methods (Pearson, Spearman, Kendall)
- **Variable Pair Scatterplots**: Explore relationships between any two variables with regression lines
- **Multiple Pair Comparison**: Compare multiple variables simultaneously with pairplots
- **Strong Correlation Detection**: Automatically identify and highlight strong correlations
- **Data Upload**: Support for CSV file uploads or use built-in sample data
- **Key Insights**: Automatic generation of correlation insights

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run correlation_dashboard.py
```

2. Open your browser to the provided URL (typically http://localhost:8501)

3. Either:
   - Click "Use Sample Data" to load demo data
   - Upload your own CSV file with numeric columns

## Dashboard Sections

### 1. Data Overview
- Dataset dimensions and basic statistics
- Option to view raw data

### 2. Correlation Heatmap
- Interactive correlation matrix visualization
- Configurable correlation methods
- Strong correlation threshold adjustment

### 3. Variable Pair Scatterplots
- Dropdown selection for X and Y variables
- Optional regression lines
- Customizable point size and transparency
- Real-time correlation calculation

### 4. Multiple Pair Comparison
- Pairwise relationship visualization
- Correlation matrix for selected variables

### 5. Key Insights
- Automatic identification of strongest correlations
- Summary of correlation patterns
- Most connected variables

## Sample Data

The dashboard includes sample sales data with variables like:
- Sales amount
- Marketing spend
- Customer satisfaction
- Product price
- Discount percentage
- Temperature
- Revenue

## Supported File Formats

- CSV files with numeric columns
- Headers should be in the first row

## Libraries Used

- **Streamlit**: Web app framework
- **Pandas**: Data manipulation
- **Seaborn**: Statistical visualization
- **Matplotlib**: Plotting backend
- **NumPy**: Numerical operations# Simple-Correlation-Analysis-Dashboard
