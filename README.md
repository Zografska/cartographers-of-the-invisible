# Cartographers of the Invisible 🗺️

A Python package for data analysis and visualization, with an interactive Streamlit web application for exploring your data.

## Features

- 📊 **Python Modules**: Reusable utilities for data loading, processing, and visualization
- 🎨 **Interactive Visualizations**: Create beautiful charts using Matplotlib and Plotly
- 🚀 **Streamlit App**: Web-based interface for exploring and visualizing data
- 📓 **Jupyter Integration**: Easy to use in notebooks for analysis workflows

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
└── README.md                     # This file
```

## Module Documentation

### data_utils

Functions for loading and processing data:

- `load_data(filepath, **kwargs)`: Load data from various file formats (CSV, JSON, Excel, Parquet)
- `process_data(df, columns=None, dropna=False, normalize=False)`: Clean and transform data
- `get_summary_statistics(df)`: Get comprehensive statistics about your dataset

### visualization

Functions for creating visualizations:

- `create_plot(df, x, y, kind='scatter', **kwargs)`: Create matplotlib plots
- `create_interactive_chart(df, x, y, kind='scatter', **kwargs)`: Create interactive Plotly charts
- `create_heatmap(df, **kwargs)`: Create correlation heatmaps

## Examples

### Load and Process Data

```python
from cartographers.data_utils import load_data, process_data

# Load data
df = load_data('mydata.csv')

# Process: select columns, drop missing values, normalize
processed_df = process_data(
    df,
    columns=['col1', 'col2', 'col3'],
    dropna=True,
    normalize=True
)
```

### Create Visualizations

```python
from cartographers.visualization import create_interactive_chart, create_heatmap

# Interactive scatter plot
fig = create_interactive_chart(
    df,
    x='temperature',
    y='humidity',
    kind='scatter',
    color='category',
    title='Temperature vs Humidity'
)
fig.show()

# Correlation heatmap
fig = create_heatmap(df, title='Feature Correlations')
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

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Authors

Cartographers Team

## Support

If you encounter any issues or have questions, please open an issue on GitHub.