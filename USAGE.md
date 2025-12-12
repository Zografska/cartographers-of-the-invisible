# Usage Guide 📖

> Technical documentation for the Cartographers of the Invisible toolkit

## Overview

This document provides technical documentation for using the Python modules and tools that have been scaffolded for the Cartographers of the Invisible project. These tools provide foundational capabilities for data loading, processing, and visualization that can be extended for LLM embedding analysis and semantic space exploration.

## Installation

### From Source

Clone the repository and install the package:

```bash
git clone https://github.com/Zografska/cartographers-of-the-invisible.git
cd cartographers-of-the-invisible
pip install -e .
```

### Install Dependencies

To install all dependencies including the Streamlit app:

```bash
pip install -r requirements.txt
```

Or install specific components:

```bash
# Core package only
pip install -e .

# With Streamlit app
pip install -e ".[app]"

# With development tools
pip install -e ".[dev]"
```

## Project Scaffolding

The current repository contains the foundational scaffolding for the project, including:

- 📊 **Python Modules**: Reusable utilities for data loading, processing, and visualization
- 🎨 **Interactive Visualizations**: Tools for creating charts using Matplotlib and Plotly
- 🚀 **Streamlit App**: Web-based interface for exploring and visualizing data
- 📓 **Jupyter Integration**: Example notebooks for analysis workflows

These tools are designed to be extended and adapted for embedding analysis, semantic space visualization, and LLM interpretability research.

## Quick Start

### Using Python Modules

The package provides utilities for data analysis and visualization:

```python
from cartographers.data_utils import load_data, process_data, get_summary_statistics
from cartographers.visualization import create_plot, create_interactive_chart

# Load and process data
df = load_data('data.csv')
processed_df = process_data(df, normalize=True)

# Create visualizations
fig = create_interactive_chart(df, x='column1', y='column2', kind='scatter')
fig.show()
```

### Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` and allows you to:
- Upload your own data files (CSV, Excel, JSON, Parquet)
- Explore data with interactive visualizations
- Generate summary statistics
- Filter and transform data
- Download processed data

### Using Jupyter Notebooks

Check out the example notebook in the `notebooks/` directory:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

## Project Structure

```
cartographers-of-the-invisible/
├── src/
│   └── cartographers/
│       ├── __init__.py          # Package initialization
│       ├── data_utils.py        # Data loading and processing utilities
│       └── visualization.py     # Visualization functions
├── notebooks/
│   └── example_usage.ipynb      # Example Jupyter notebook
├── app.py                        # Streamlit web application
├── setup.py                      # Package setup configuration
├── requirements.txt              # Project dependencies
├── README.md                     # Project overview
└── USAGE.md                      # This file
```

## Module Documentation

### data_utils

Functions for loading and processing data:

#### `load_data(filepath, **kwargs)`

Load data from various file formats.

**Parameters:**
- `filepath` (str): Path to the data file
- `**kwargs`: Additional arguments to pass to pandas read functions

**Supported formats:**
- CSV (`.csv`)
- JSON (`.json`)
- Excel (`.xls`, `.xlsx`)
- Parquet (`.parquet`)

**Returns:**
- `pd.DataFrame`: DataFrame containing the loaded data

**Example:**
```python
from cartographers.data_utils import load_data

df = load_data('embeddings.csv')
```

#### `process_data(df, columns=None, dropna=False, normalize=False)`

Process and clean data.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `columns` (list, optional): List of columns to keep (None keeps all)
- `dropna` (bool): Whether to drop rows with missing values
- `normalize` (bool): Whether to normalize numeric columns

**Returns:**
- `pd.DataFrame`: Processed DataFrame

**Example:**
```python
from cartographers.data_utils import process_data

processed_df = process_data(
    df,
    columns=['dim1', 'dim2', 'dim3'],
    dropna=True,
    normalize=True
)
```

#### `get_summary_statistics(df)`

Get summary statistics for a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- `dict`: Dictionary containing summary statistics including shape, columns, dtypes, missing values, and numeric summaries

**Example:**
```python
from cartographers.data_utils import get_summary_statistics

stats = get_summary_statistics(df)
print(f"Dataset shape: {stats['shape']}")
print(f"Missing values: {stats['missing_values']}")
```

### visualization

Functions for creating visualizations:

#### `create_plot(df, x, y=None, kind='scatter', title=None, figsize=(10, 6), **kwargs)`

Create a matplotlib plot.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `x` (str): Column name for x-axis
- `y` (str, optional): Column name for y-axis (not used for 'hist')
- `kind` (str): Type of plot ('scatter', 'line', 'bar', 'hist')
- `title` (str, optional): Plot title
- `figsize` (tuple): Figure size (width, height)
- `**kwargs`: Additional arguments for plotting

**Returns:**
- `matplotlib.figure.Figure`: Matplotlib Figure object

**Example:**
```python
from cartographers.visualization import create_plot

fig = create_plot(
    df,
    x='dimension1',
    y='dimension2',
    kind='scatter',
    title='Embedding Space',
    alpha=0.6
)
fig.show()
```

#### `create_interactive_chart(df, x, y=None, kind='scatter', title=None, color=None, **kwargs)`

Create an interactive Plotly chart.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `x` (str): Column name for x-axis
- `y` (str, optional): Column name for y-axis (not used for 'hist')
- `kind` (str): Type of chart ('scatter', 'line', 'bar', 'hist')
- `title` (str, optional): Chart title
- `color` (str, optional): Column name for color encoding
- `**kwargs`: Additional arguments for Plotly

**Returns:**
- `plotly.graph_objects.Figure`: Plotly Figure object

**Example:**
```python
from cartographers.visualization import create_interactive_chart

fig = create_interactive_chart(
    df,
    x='umap1',
    y='umap2',
    kind='scatter',
    color='cluster',
    title='UMAP Projection of Embeddings',
    hover_data=['word', 'category']
)
fig.show()
```

#### `create_heatmap(df, title=None, **kwargs)`

Create a correlation heatmap for numeric columns.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `title` (str, optional): Chart title
- `**kwargs`: Additional arguments for Plotly

**Returns:**
- `plotly.graph_objects.Figure`: Plotly Figure object

**Example:**
```python
from cartographers.visualization import create_heatmap

fig = create_heatmap(df, title='Feature Correlations')
fig.show()
```

## Examples

### Load and Process Embedding Data

```python
from cartographers.data_utils import load_data, process_data

# Load embedding data
embeddings_df = load_data('llm_embeddings.csv')

# Process: select relevant dimensions, normalize
processed_embeddings = process_data(
    embeddings_df,
    columns=['dim1', 'dim2', 'dim3', 'label'],
    dropna=True,
    normalize=True
)
```

### Create Visualizations for Embeddings

```python
from cartographers.visualization import create_interactive_chart, create_heatmap

# Interactive scatter plot of embedding space
fig = create_interactive_chart(
    processed_embeddings,
    x='dim1',
    y='dim2',
    kind='scatter',
    color='label',
    title='Embedding Space Visualization',
    hover_data=['word']
)
fig.show()

# Correlation heatmap
fig = create_heatmap(processed_embeddings, title='Dimension Correlations')
fig.show()
```

### Using the Streamlit App

The included Streamlit app provides an interactive interface for:

1. **Data Upload**: Upload CSV, Excel, JSON, or Parquet files
2. **Visualization**: Create scatter plots, line charts, and heatmaps
3. **Filtering**: Select specific columns and apply transformations
4. **Export**: Download processed data

Run the app:
```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Extending the Toolkit

This scaffolding is designed to be extended for LLM-specific functionality:

### Future Enhancements

1. **Embedding Extraction**: Add modules to extract embeddings from LLMs (e.g., using HuggingFace transformers)

2. **Dimensionality Reduction**: Integrate UMAP, t-SNE, or PCA for high-dimensional embedding visualization

3. **Clustering**: Add clustering algorithms (K-means, DBSCAN, etc.) for semantic grouping

4. **Model Integration**: Connect to LLM APIs for real-time embedding generation

5. **Advanced Visualizations**: Add 3D visualizations, animated plots, and interactive semantic maps

### Example Extension: Adding UMAP

```python
# Example of how you might extend the toolkit
import umap
import pandas as pd
from cartographers.data_utils import load_data
from cartographers.visualization import create_interactive_chart

# Load high-dimensional embeddings
embeddings = load_data('bert_embeddings.csv')

# Separate features and labels
feature_cols = [col for col in embeddings.columns if col != 'label']
X = embeddings[feature_cols]
labels = embeddings['label']

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(X)

# Create DataFrame with reduced dimensions
reduced_df = pd.DataFrame(
    embedding_2d,
    columns=['umap1', 'umap2']
)
reduced_df['label'] = labels

# Visualize
fig = create_interactive_chart(
    reduced_df,
    x='umap1',
    y='umap2',
    kind='scatter',
    color='label',
    title='UMAP Projection of BERT Embeddings'
)
fig.show()
```

## Requirements

- Python >= 3.8
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- plotly >= 5.14.0
- streamlit >= 1.28.0 (for the web app)
- openpyxl >= 3.1.0 (for Excel support)

## Contributing

Contributions to extend this toolkit are welcome! Please feel free to:
- Add new visualization methods
- Implement embedding extraction utilities
- Enhance the Streamlit app with new features
- Submit example notebooks

## Support

If you encounter any issues or have questions:
1. Check this documentation
2. Review the example notebook in `notebooks/example_usage.ipynb`
3. Open an issue on GitHub

## License

This project is open source and available under the MIT License.

---

**For project overview and research goals, see [README.md](README.md).**