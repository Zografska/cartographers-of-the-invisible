"""
Visualization utilities for creating plots and charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple


def create_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = 'scatter',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> plt.Figure:
    """
    Create a matplotlib plot.
    
    Args:
        df: Input DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        kind: Type of plot ('scatter', 'line', 'bar', 'hist')
        title: Plot title
        figsize: Figure size (width, height)
        **kwargs: Additional arguments for plotting
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind == 'scatter':
        ax.scatter(df[x], df[y], **kwargs)
    elif kind == 'line':
        ax.plot(df[x], df[y], **kwargs)
    elif kind == 'bar':
        ax.bar(df[x], df[y], **kwargs)
    elif kind == 'hist':
        ax.hist(df[x], **kwargs)
    else:
        raise ValueError(f"Unsupported plot type: {kind}")
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig


def create_interactive_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = 'scatter',
    title: Optional[str] = None,
    color: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create an interactive Plotly chart.
    
    Args:
        df: Input DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        kind: Type of chart ('scatter', 'line', 'bar', 'histogram')
        title: Chart title
        color: Column name for color encoding
        **kwargs: Additional arguments for Plotly
        
    Returns:
        Plotly Figure object
    """
    if kind == 'scatter':
        fig = px.scatter(df, x=x, y=y, color=color, title=title, **kwargs)
    elif kind == 'line':
        fig = px.line(df, x=x, y=y, color=color, title=title, **kwargs)
    elif kind == 'bar':
        fig = px.bar(df, x=x, y=y, color=color, title=title, **kwargs)
    elif kind == 'histogram':
        fig = px.histogram(df, x=x, color=color, title=title, **kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {kind}")
    
    return fig


def create_heatmap(
    df: pd.DataFrame,
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Input DataFrame
        title: Chart title
        **kwargs: Additional arguments for Plotly
        
    Returns:
        Plotly Figure object
    """
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        title=title or 'Correlation Heatmap',
        color_continuous_scale='RdBu',
        aspect='auto',
        **kwargs
    )
    
    return fig
