import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Correlation Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Simple Correlation Analysis Dashboard")
st.markdown("Explore correlations between variables in your numeric dataset")

# Sidebar for file upload and options
st.sidebar.header("Data Input")

# Sample data option
if st.sidebar.button("Use Sample Data"):
    # Generate sample sales data
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'sales_amount': np.random.normal(5000, 1500, n_samples),
        'marketing_spend': np.random.normal(1000, 300, n_samples),
        'customer_satisfaction': np.random.uniform(1, 10, n_samples),
        'product_price': np.random.normal(50, 15, n_samples),
        'discount_percent': np.random.uniform(0, 30, n_samples),
        'temperature': np.random.normal(20, 10, n_samples)
    }
    
    # Add some correlations
    data['sales_amount'] = data['sales_amount'] + 0.7 * data['marketing_spend'] + 0.5 * data['customer_satisfaction']
    data['customer_satisfaction'] = data['customer_satisfaction'] + 0.3 * (100 - data['product_price'])
    
    df = pd.DataFrame(data)
    st.session_state['df'] = df
    st.sidebar.success("Sample data loaded!")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with numeric columns"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.sidebar.success(f"File uploaded! Shape: {df.shape}")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Check if data is available
if 'df' not in st.session_state:
    st.info("👆 Please upload a CSV file or use sample data from the sidebar to get started")
    st.stop()

df = st.session_state['df']

# Data overview
st.header("📋 Data Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

# Show first few rows
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))

# Filter numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.empty:
    st.error("No numeric columns found in the dataset!")
    st.stop()

# Correlation Analysis Section
st.header("🔥 Correlation Heatmap")

# Heatmap options
col1, col2 = st.columns(2)
with col1:
    correlation_method = st.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"],
        help="Choose correlation calculation method"
    )

with col2:
    show_values = st.checkbox("Show correlation values", value=True)

# Calculate correlation matrix
correlation_matrix = numeric_df.corr(method=correlation_method)

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=show_values,
    cmap='RdBu_r',
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={"shrink": .8},
    ax=ax
)

ax.set_title(f'{correlation_method.title()} Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
st.pyplot(fig)

# Strong correlations summary
st.subheader("🎯 Strong Correlations")
threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.7, 0.05)

# Find strong correlations
strong_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) >= threshold:
            strong_corr.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlation': corr_val,
                'Strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
            })

if strong_corr:
    strong_corr_df = pd.DataFrame(strong_corr)
    strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
    st.dataframe(strong_corr_df, use_container_width=True)
else:
    st.info(f"No correlations found above threshold {threshold}")# Scatterplot Analysis Section
st.header("📈 Variable Pair Scatterplots")

# Column selection for scatterplot
col1, col2 = st.columns(2)

with col1:
    x_variable = st.selectbox(
        "Select X variable",
        numeric_df.columns,
        key="x_var"
    )

with col2:
    y_variable = st.selectbox(
        "Select Y variable",
        numeric_df.columns,
        index=1 if len(numeric_df.columns) > 1 else 0,
        key="y_var"
    )

# Scatterplot options
col1, col2, col3 = st.columns(3)

with col1:
    show_regression = st.checkbox("Show regression line", value=True)

with col2:
    point_size = st.slider("Point size", 10, 100, 50)

with col3:
    alpha = st.slider("Point transparency", 0.1, 1.0, 0.7)

# Create scatterplot
fig, ax = plt.subplots(figsize=(10, 6))

# Basic scatter plot
ax.scatter(
    numeric_df[x_variable], 
    numeric_df[y_variable], 
    alpha=alpha, 
    s=point_size,
    color='steelblue'
)

# Add regression line if requested
if show_regression:
    sns.regplot(
        x=x_variable, 
        y=y_variable, 
        data=numeric_df, 
        scatter=False, 
        color='red', 
        ax=ax
    )

# Calculate and display correlation
corr_value = numeric_df[x_variable].corr(numeric_df[y_variable])
ax.set_title(f'{x_variable} vs {y_variable}\nCorrelation: {corr_value:.3f}', fontsize=14)
ax.set_xlabel(x_variable, fontsize=12)
ax.set_ylabel(y_variable, fontsize=12)

plt.tight_layout()
st.pyplot(fig)

# Correlation statistics
st.subheader("📊 Pair Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Correlation", f"{corr_value:.3f}")

with col2:
    st.metric("X Mean", f"{numeric_df[x_variable].mean():.2f}")

with col3:
    st.metric("Y Mean", f"{numeric_df[y_variable].mean():.2f}")

with col4:
    st.metric("Data Points", len(numeric_df))

# Multiple pair comparison
st.header("🔍 Multiple Pair Comparison")

# Select multiple variables for comparison
selected_vars = st.multiselect(
    "Select variables for pairwise comparison",
    numeric_df.columns,
    default=list(numeric_df.columns[:4]) if len(numeric_df.columns) >= 4 else list(numeric_df.columns)
)

if len(selected_vars) >= 2:
    # Create pairplot
    fig = plt.figure(figsize=(12, 10))
    
    # Use seaborn pairplot
    pair_df = numeric_df[selected_vars]
    g = sns.pairplot(pair_df, diag_kind='hist', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pairwise Variable Relationships', y=1.02, fontsize=16)
    
    st.pyplot(g.fig)
    
    # Show correlation matrix for selected variables
    st.subheader("Correlation Matrix (Selected Variables)")
    selected_corr = pair_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        selected_corr,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.3f',
        ax=ax
    )
    ax.set_title('Selected Variables Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

# Data insights
st.header("💡 Key Insights")

# Generate automatic insights
insights = []

# Find highest correlation
max_corr = 0
max_pair = None
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > max_corr:
            max_corr = corr_val
            max_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])

if max_pair:
    insights.append(f"🔥 Strongest correlation: {max_pair[0]} and {max_pair[1]} ({max_corr:.3f})")

# Count strong correlations
strong_count = len([1 for i in range(len(correlation_matrix.columns)) 
                   for j in range(i+1, len(correlation_matrix.columns))
                   if abs(correlation_matrix.iloc[i, j]) >= 0.7])

insights.append(f"📊 Found {strong_count} strong correlations (|r| ≥ 0.7)")

# Variable with most correlations
var_corr_counts = {}
for col in correlation_matrix.columns:
    count = sum(1 for other_col in correlation_matrix.columns 
                if col != other_col and abs(correlation_matrix.loc[col, other_col]) >= 0.5)
    var_corr_counts[col] = count

if var_corr_counts:
    most_connected = max(var_corr_counts, key=var_corr_counts.get)
    insights.append(f"🌐 Most connected variable: {most_connected} ({var_corr_counts[most_connected]} moderate+ correlations)")

for insight in insights:
    st.write(insight)

