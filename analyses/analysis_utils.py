import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional
from kneed import KneeLocator
from scipy.stats import pearsonr
import plotly.graph_objects as go


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


def analyze_components_correlation(df: pd.DataFrame, min_repos: int = 20, p_value_threshold: float = 0.05) -> pd.DataFrame:
    """
    Analyzes the correlation between component usage and repository size for components
    that appear in a minimum number of repositories and returns the significant correlations.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data. 
                       Expected columns: 'repo', 'component_name', 'count', 'size', 'component_type'.
    min_repos (int): Minimum number of repositories a component must appear in to be considered. Default is 20.
    p_value_threshold (float): The p-value threshold for determining significant correlations. Default is 0.05.

    Returns:
    pd.DataFrame: A DataFrame containing significant correlations sorted by correlation coefficient in descending order.
                  Columns: 'component_type', 'component_name', 'correlation', 'p_value', 'original_num_repos', 'num_usages'.
    """
    component_repo_count = df.groupby('component_name')['repo'].nunique()
    filtered_components = component_repo_count[component_repo_count >= min_repos].index
    pivot_df = df.pivot_table(index='repo', columns='component_name', values='count', aggfunc='sum', fill_value=0)
    pivot_df = pivot_df.reindex(columns=filtered_components, fill_value=0).reset_index()
    complete_df = pivot_df.melt(id_vars='repo', var_name='component_name', value_name='total_usage')
    repo_sizes = df[['repo', 'size']].drop_duplicates()
    complete_df = complete_df.merge(repo_sizes, on='repo', how='left')
    complete_df.rename(columns={'size': 'repo_size'}, inplace=True)
    component_types = df[['component_name', 'component_type']].drop_duplicates()
    complete_df = complete_df.merge(component_types, on='component_name', how='left')

    correlations = []
    for component in filtered_components:
        comp_df = complete_df[complete_df['component_name'] == component]
        if len(comp_df) > 1:
            corr, p_value = pearsonr(comp_df['total_usage'], comp_df['repo_size'])
            total_usages = comp_df['total_usage'].sum()
            original_num_repos = df[df['component_name'] == component]['repo'].nunique()
            component_type = comp_df['component_type'].iloc[0]
            correlations.append([component_type, component, corr, p_value, original_num_repos, total_usages])
    correlation_df = pd.DataFrame(correlations, columns=['component_type', 'component_name', 'correlation', 'p_value', 'original_num_repos', 'num_usages'])
    significant_correlations = correlation_df[correlation_df['p_value'] < p_value_threshold]
    sorted_correlations = significant_correlations.sort_values(by='correlation', ascending=False)

    return sorted_correlations


def identify_elbow_components(df: pd.DataFrame, component_column: str = 'component_name', usage_column: str = 'count') -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Identifies the components to the right of the "elbow" in the Lorenz Curve.

    Parameters:
    df: DataFrame containing the data.
    component_column: The name of the column containing component names. Default is 'component_name'.
    usage_column: The name of the column containing usage counts. Default is 'count'.

    Returns:
    elbow_components: DataFrame of components to the right of the elbow point, including component_type.
    elbow_index: The index of the elbow point.
    component_usage: DataFrame containing cumulative components and usages for all components.
    """
    component_usage = df.groupby(['component_type', component_column])[usage_column].sum().reset_index()
    component_usage = component_usage.sort_values(by=usage_column)
    component_usage['cum_components'] = np.arange(1, len(component_usage) + 1) / len(component_usage)
    component_usage['cum_usages'] = component_usage[usage_column].cumsum() / component_usage[usage_column].sum()
    kneedle = KneeLocator(component_usage['cum_components'], component_usage['cum_usages'], curve='convex', direction='increasing')
    elbow_index = kneedle.elbow
    elbow_components = component_usage[component_usage['cum_components'] > elbow_index]
    
    return elbow_components, elbow_index, component_usage


def plot_lorenz_curve(component_usage: pd.DataFrame, elbow_index: float) -> None:
    """
    Plots the Lorenz Curve for component usages and identifies the elbow point.

    Parameters:
    component_usage: DataFrame with cumulative component usage data.
    elbow_index: The index of the elbow point.
    """
    plt.figure(figsize=(18, 8))
    plt.plot(component_usage['cum_components'], component_usage['cum_usages'], marker='o', color=COLOR_MAP['function'], label='Lorenz Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color=COLOR_MAP['exception'], label='Line of Equality')
    plt.axvline(elbow_index, color=COLOR_MAP['class'], linestyle='--', label='Elbow Point')
    plt.xlabel('Cumulative Percentage of Components')
    plt.ylabel('Cumulative Percentage of Usages')
    plt.title('Lorenz Curve of Component Usages')
    plt.legend()
    plt.show()


def create_styled_table(df: pd.DataFrame, output_filename: str = "styled_df_image.png") -> None:
    """
    Generate a styled table using Plotly based on a dataframe and a color mapping.

    Parameters:
    - df (pd.DataFrame): A dataframe with the columns 'component_type', 'cum_components', 'cum_usages'.
    - output_filename (str): Path to save the output image file.
    """
    df = df.copy()
    df['color'] = df['component_type'].map(COLOR_MAP)
    df['cum_components'] = df['cum_components'].map(lambda x: f"{x:.3f}")
    df['cum_usages'] = df['cum_usages'].map(lambda x: f"{x:.3f}")

    text_colors = [
        [
            'white' if _luminance(COLOR_MAP[ct]) < 0.5 else 'black' 
            for ct in df['component_type']
        ]
        for _ in df.columns[:-1]
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{col}</b>" for col in df.columns[:-1]],
            fill_color='#f7f7f9',
            font=dict(color='#333', size=12, family="Arial"),
            align='left',
            line_color='#f7f7f9'
        ),
        cells=dict(
            values=[df[col] for col in df.columns[:-1]],
            fill_color=[df['color']] * len(df.columns[:-1]),
            font=dict(color=[text_colors[i] for i in range(len(df.columns[:-1]))], size=12),
            align='left',
            line_color=[df['color']] * len(df.columns[:-1])
        )
    )])

    fig.update_layout(
        height=40 + 20 * len(df),
        margin=dict(l=5, r=5, t=5, b=5)
    )
    fig.write_image(output_filename)
    fig.show()
