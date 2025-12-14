"""
Streamlit App for Cartographers of the Invisible

A web application for visualizing and exploring data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


st.set_page_config(
    page_title="Cartographers of the Invisible",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗺️ Cartographers of the Invisible")
st.markdown("### Data Visualization and Exploration Tool")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Upload File", "Generate Random Data"]
    )
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app helps you visualize and explore your data.")

# Main content
def load_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'value_a': np.cumsum(np.random.randn(100)) + 100,
        'value_b': np.cumsum(np.random.randn(100)) + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'metric': np.random.randint(1, 100, 100)
    }
    return pd.DataFrame(data)

def generate_random_data(rows=100):
    """Generate random data"""
    np.random.seed(None)
    data = {
        'x': np.random.randn(rows),
        'y': np.random.randn(rows),
        'z': np.random.randint(0, 10, rows),
        'category': np.random.choice(['Cat1', 'Cat2', 'Cat3'], rows)
    }
    return pd.DataFrame(data)

# Data loading
df = None

if data_source == "Sample Data":
    df = load_sample_data()
    st.success("✅ Sample data loaded successfully!")
    
elif data_source == "Upload File":
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet']
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            
            st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a file to continue")
        
elif data_source == "Generate Random Data":
    num_rows = st.sidebar.slider("Number of rows", 10, 1000, 100)
    df = generate_random_data(num_rows)
    st.success(f"✅ Generated {num_rows} rows of random data!")

# Display data and visualizations
if df is not None:
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    with st.expander("🔍 Preview Data", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Data summary
    with st.expander("📈 Summary Statistics"):
        st.write(df.describe())
    
    # Visualizations
    st.header("📉 Visualizations")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) >= 2:
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Line Chart", "Correlation Heatmap"])
        
        with tab1:
            st.subheader("Scatter Plot")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key='scatter_x')
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key='scatter_y')
            
            color_col = st.selectbox("Color by", [None] + all_cols, key='scatter_color')
            
            try:
                from cartographers.utils.visualization import create_interactive_chart
                fig = create_interactive_chart(
                    df,
                    x=x_col,
                    y=y_col,
                    kind='scatter',
                    color=color_col,
                    title=f"{y_col} vs {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                import plotly.express as px
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Line Chart")
            x_col_line = st.selectbox("X-axis", all_cols, key='line_x')
            y_col_line = st.selectbox("Y-axis", numeric_cols, key='line_y')
            color_col_line = st.selectbox("Color by", [None] + all_cols, key='line_color')
            
            try:
                from cartographers.utils.visualization import create_interactive_chart
                fig = create_interactive_chart(
                    df,
                    x=x_col_line,
                    y=y_col_line,
                    kind='line',
                    color=color_col_line,
                    title=f"{y_col_line} over {x_col_line}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                import plotly.express as px
                fig = px.line(df, x=x_col_line, y=y_col_line, color=color_col_line,
                            title=f"{y_col_line} over {x_col_line}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Heatmap")
            if len(numeric_cols) >= 2:
                try:
                    from cartographers.utils.visualization import create_heatmap
                    fig = create_heatmap(df)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    import plotly.express as px
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title='Correlation Heatmap',
                                  color_continuous_scale='RdBu',
                                  aspect='auto')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
    
    elif len(numeric_cols) >= 1:
        st.subheader("Histogram")
        col = st.selectbox("Select column", numeric_cols)
        
        import plotly.express as px
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No numeric columns found for visualization")
    
    # Data filtering
    st.header("🔧 Data Processing")
    
    with st.expander("Filter and Transform Data"):
        st.write("Select columns to keep:")
        selected_cols = st.multiselect("Columns", all_cols, default=all_cols)
        
        drop_na = st.checkbox("Drop rows with missing values")
        
        if st.button("Apply Filters"):
            filtered_df = df[selected_cols].copy()
            if drop_na:
                filtered_df = filtered_df.dropna()
            
            st.success(f"✅ Filtered data: {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
            st.dataframe(filtered_df.head(), use_container_width=True)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Please select a data source from the sidebar to begin")

# Footer
st.markdown("---")
st.markdown("**Cartographers of the Invisible** | Data Visualization Tool")
