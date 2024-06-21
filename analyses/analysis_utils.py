import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional
from matplotlib.colors import LinearSegmentedColormap


COLOR_MAP = {
    'attribute': '#488A99',
    'class': '#DBAE58',
    'exception': '#AC3E31',
    'function': '#484848',
    'method': '#DADADA',
    # 'repos': '#488A99',
    # 'files': '#484848',
    # 'corr_low': '#DADADA',
    # 'corr_high': '#484848'
}


def extract_repo_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts repository and file details from the 'filename' column.

    Parameters:
    df: DataFrame containing columns 'filename', 'library', 'component_type', 'component_name', 'count'.

    Returns:
    DataFrame with columns 'repo', 'filename', 'library', 'component_type', 'component_name', 'count'.
    """
    df = df.copy()
    df['repo'] = df['filename'].str.split('/').str[5:7].str.join('/')
    df['filename'] = df.apply(lambda row: row['filename'].split(row['repo'])[-1], axis=1).str[1:]
    df = df[['repo', 'filename', 'library', 'component_type', 'component_name', 'count']]
    return df


def count_component_occurrences(df: pd.DataFrame, within_column: Optional[str] = None) -> pd.Series:
    """
    Count the occurrences of each unique component name and type combination in a DataFrame.

    This function can count occurrences within a specified column (e.g., 'repo' or 'filename')
    or across the entire DataFrame if no column is specified.

    Parameters:
    df: The DataFrame containing the data to be analyzed. It must include the columns
        'component_name', 'component_type', and optionally the column specified in `within_column`.
    within_column: The column within which to count unique occurrences of each component name and type combination.
                   If set to None, the function counts all occurrences across the entire DataFrame. Default is None.

    Returns:
    A Series with the count of each unique component name and type combination, sorted in descending order.
    """
    if within_column is None:
        component_counts = df.groupby(['component_name', 'component_type']).size()
    else:
        unique_components = df[[within_column, 'component_type', 'component_name']].drop_duplicates()
        component_counts = unique_components.groupby(['component_name', 'component_type']).size()
    sorted_component_counts = component_counts.sort_values(ascending=False)
    return sorted_component_counts


def _luminance(color):
    """
    Calculate the luminance of a color.
    """
    rgb = mcolors.hex2color(color)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def plot_component_popularity(series: pd.Series, title: str, divide_by_to_show_prc: int = None, top_n: int = 10, bar_height: float = 0.8) -> None:
    """
    Plots the popularity of components as a horizontal bar plot.

    Parameters:
    series: A pandas Series with a MultiIndex (component_name, component_type) and integer values.
    title: The title of the plot.
    top_n: The number of top components to display. Default is 10.
    bar_height: The height of each bar in the plot. Default is 0.8.
    """
    df = series.reset_index(name='count')
    df = df.nlargest(top_n, 'count')
    df = df.sort_values('count')

    max_value = df['count'].max()
    if divide_by_to_show_prc:
        df['count'] = (df['count'] / divide_by_to_show_prc) * 100
        max_value = df['count'].max()
        # df['count'] = df['count'].apply(lambda x: f"{x:.1f}%")
    
    fig_height = top_n * bar_height + 2
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    colors = df['component_type'].map(COLOR_MAP)
    df['component_name'] = df['component_name'].str.rjust(25)
    bars = ax.barh(df['component_name'], df['count'], color=colors, edgecolor='white', height=bar_height)
    
    for index, value in enumerate(df['count']):
        bar_color = colors.iloc[index]
        text_color = 'white' if _luminance(bar_color) < 0.5 else 'black'
        ax.text(max_value/100, index, f"{value:.1f}%" if divide_by_to_show_prc else f"{value}", va='center_baseline', ha='left', fontsize=12, color=text_color, backgroundcolor=bar_color)
    
    ax.set_title(f"{title}{24*' '}", fontsize=16, loc='center')
    
    ax.tick_params(axis='x', length=0)
    ax.set_xticklabels([])
    ax.tick_params(axis='y', length=0, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in COLOR_MAP.values()]
    labels = COLOR_MAP.keys()
    ax.legend(handles, labels, title='Component Type', loc='lower right', fontsize=12, title_fontsize=12, bbox_to_anchor=(0.97, 0.03))
    
    plt.show()
