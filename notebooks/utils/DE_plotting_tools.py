
import pandas as pd
import numpy as np 
import matplotlib as mpl

import plotly.express as px 
import plotly.graph_objects as go

import umap
from sklearn.decomposition import PCA
from scipy import stats
import random

def compare_vol_plot(DE_df_list: list, 
                     DE_df_name: list, 
                     fig_title = '', 
                     fig_dimension = None, 
                     fig_fixed_range = False, 
                     FDR_line = 0.05):
    
    assert len(DE_df_list) == len(DE_df_name), print("Please ensure DE_df_list and DE_df_name len matches")
    
    # Initialize ranges for plots to prevent autosizing
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    
    fig = go.Figure()
    # Add traces of individual DE_df 
    for DE_df, DE_name in zip(DE_df_list, DE_df_name): 
        plot_df = DE_df.copy()
        fig.add_trace(go.Scatter(x = plot_df['logFC'], 
                                y = -np.log10(plot_df['FDR']),
                                text = plot_df['symbol'],
                                mode = 'markers', 
                                name = DE_name,
                                marker = dict(size = 10, 
                                            #    color = 'grey', 
                                            opacity=0.3)
                                )
                    )
        if fig_fixed_range:
            xmin = min(plot_df['logFC'])*1.10 if min(plot_df['logFC']) < xmin else xmin
            xmax = max(plot_df['logFC'])*1.10 if max(plot_df['logFC']) > xmax else xmax
            # ymin =  min(-np.log10(plot_df['FDR']))*1.10 if min(-np.log10(plot_df['FDR'])) < ymin else ymin  # ymin will be 0 anyways
            ymax =  max(-np.log10(plot_df['FDR']))*1.10 if max(-np.log10(plot_df['FDR'])) > ymax else ymax


    # Add a line for FDR = 0.05
    fig.add_shape(type='line', x0=-10, x1=10,
                  y0=-np.log10(FDR_line), y1=-np.log10(FDR_line),
                  line=dict(color='violet', width=3, dash='dash'))

    fig.update_traces( 
        textposition='top center',
        hovertemplate ='<b>%{text}</b>' + '<br>LogFC: %{x}' + '<br>FDR: %{y}<br>')
    
    # Define ranges to avoid autoresizing when hiding data
    if fig_fixed_range:
        # Center the data by taking the bigger value between xmin and xmax 
        xmax = max(abs(xmin),abs(xmax))
        fig.update_xaxes(range=[-xmax, xmax])
        fig.update_yaxes(range=[-1, ymax])

    if fig_dimension is None:
        fig.update_layout(
            title=fig_title,
            autosize=True,
            template='simple_white'
        )
    else: 
        fig.update_layout(
            title=fig_title,
            width=fig_dimension[0],
            height=fig_dimension[1],
            template='simple_white'
        )
    return fig 

def vol_plot(DE_df: pd.DataFrame, 
             logFC_group = ['WT', 'KO'],
             interested_genes = None, 
             manual_color = None,
             fig_title = '', 
             fig_dimension = None, 
             fig_fixed_range = False, 
             FDR_cutoff = None, 
             FDR_line = 0.05, 
             plot_none_sig = True, 
             opacity=0.3, 
             ymin = -0.25):
        
    # Initialize ranges for plots to prevent autosizing
    xmin, xmax, ymax = 0, 0, 0
    if manual_color is None: manual_color = distinct_colors(logFC_group)
    
    fig = go.Figure()
    # Add traces of individual DE_df 
    plot_df = DE_df.copy()
    
    if interested_genes: 
        # FDR_cutoff = 0.05 if FDR_cutoff is None else FDR_cutoff
        if plot_none_sig:
            # Plot non-sig first
            temp_df = plot_df.copy()
            if FDR_cutoff:  
                temp_df = plot_df[plot_df.FDR > FDR_cutoff]
            temp_df = temp_df[~temp_df.symbol.isin(interested_genes)]
            fig.add_trace(go.Scatter(x = temp_df['logFC'], 
                                    y = -np.log10(temp_df['FDR']),
                                    text = temp_df['symbol'],
                                    mode = 'markers', 
                                    name = 'na',
                                    marker = dict(size = 10, 
                                                color = manual_color['na'] if 'na' in manual_color.keys() else 'lightgrey', 
                                                opacity=0.2)
                                    )
                        )
        for _gene in interested_genes: 
            subset_df =  plot_df[plot_df.symbol == _gene]
            fig.add_trace(go.Scatter(x = subset_df['logFC'], 
                                            y = -np.log10(subset_df['FDR']),
                                            text = subset_df['symbol'],
                                            mode = 'markers', 
                                            name = _gene,
                                            marker = dict(size = 10, 
                                                        opacity=opacity)
                                            )
                                )

    else: 
        for _group in logFC_group:
            # Subset by left or right group
            subset_df =  plot_df[plot_df.logFC <= 0] if _group == logFC_group[0] else plot_df[plot_df.logFC >= 0]
            # Subset by FDR if declared
            
            if FDR_cutoff: 
                # Plot non-sig first 
                if plot_none_sig:
                    temp_df = subset_df[subset_df.FDR > FDR_cutoff]
                    fig.add_trace(go.Scatter(x = temp_df['logFC'], 
                                            y = -np.log10(temp_df['FDR']),
                                            text = temp_df['symbol'],
                                            mode = 'markers', 
                                            name = _group,
                                            marker = dict(size = 10, 
                                                        color = manual_color['na'] if 'na' in manual_color.keys() else 'lightgrey', 
                                                        opacity=opacity)
                                            )
                                )
                
                temp_df = subset_df[subset_df.FDR <= FDR_cutoff]
                fig.add_trace(go.Scatter(x = temp_df['logFC'], 
                                        y = -np.log10(temp_df['FDR']),
                                        text = temp_df['symbol'],
                                        mode = 'markers', 
                                        name = _group,
                                        marker = dict(size = 10, 
                                                    color = manual_color[_group], 
                                                    opacity=opacity)
                                        )
                            )
            else: 
                fig.add_trace(go.Scatter(x = subset_df['logFC'], 
                                        y = -np.log10(subset_df['FDR']),
                                        text = subset_df['symbol'],
                                        mode = 'markers', 
                                        name = _group,
                                        marker = dict(size = 10, 
                                                    opacity=opacity)
                                        )
                            )
    
    if fig_fixed_range:
        xmin = min(plot_df['logFC'])*1.10 if min(plot_df['logFC']) < xmin else xmin
        xmax = max(plot_df['logFC'])*1.10 if max(plot_df['logFC']) > xmax else xmax
        # ymin =  min(-np.log10(plot_df['FDR']))*1.10 if min(-np.log10(plot_df['FDR'])) < ymin else ymin  # ymin will be 0 anyways
        ymax =  max(-np.log10(plot_df['FDR']))*1.10 if max(-np.log10(plot_df['FDR'])) > ymax else ymax


    # Add a line for FDR = 0.05
    if FDR_line: 
        fig.add_shape(type='line', x0=-10, x1=10,
                    y0=-np.log10(FDR_line), y1=-np.log10(FDR_line),
                    line=dict(color='violet', width=3, dash='dash'))

    fig.update_traces( 
        textposition='top center',
        hovertemplate ='<b>%{text}</b>' + '<br>LogFC: %{x}' + '<br>FDR: %{y}<br>')
    
    # Define ranges to avoid autoresizing when hiding data
    if fig_fixed_range:
        # Center the data by taking the bigger value between xmin and xmax 
        xmax = max(abs(xmin),abs(xmax))
        fig.update_xaxes(range=[-xmax, xmax])
        fig.update_yaxes(range=[ymin, ymax])

    if fig_dimension is None:
        fig.update_layout(
            title=fig_title,
            autosize=True,
            template='simple_white'
        )
    else: 
        fig.update_layout(
            title=fig_title,
            width=fig_dimension[0],
            height=fig_dimension[1],
            template='simple_white'
        )
    return fig 


def reduced_dimension_plot(count_df: pd.DataFrame(), 
                           reduction_method = 'umap',
                           pca_n_components = 2):
    
    assert (reduction_method == 'umap') | (reduction_method == 'pca'), print('reduction_method needs to be either \'pca\' or \'umap\'')
    
    if reduction_method.lower() == 'pca':
        pca = PCA(n_components=pca_n_components)  # You can adjust the number of components
        result = pca.fit_transform(count_df.transpose())
        col = [f"pca_{i}"  for i in range(1,pca_n_components+1)]
    elif reduction_method.lower() == 'umap':
        result = umap.UMAP().fit_transform(count_df.transpose())
        col = ['umap_1', 'umap_2']
    
    result = pd.DataFrame(result, columns = col)
    result['sample_name'] = count_df.columns
    result['sample'] = result['sample_name'].str.split('_').str[0]
    result['odor'] = result['sample_name'].str.split('_').str[1]

    fig = px.scatter(result, 
                    x = col[0], 
                    y = col[1], 
                    hover_name = 'sample_name', 
                    color = 'sample', 
                    symbol = 'odor')
    
    fig.update_traces(marker={'size': 15})
    
    fig.update_layout(
        title=reduction_method,
        autosize=True,
        template='simple_white'
    )
    
    return fig


def downsample_fig(fig: go.Figure,
                   sample_method='random_sample',
                   seed = 0,
                   max_points: int = None, max_points_pct: float = None) -> go.Figure:
    """
    # TODO The function is written for violin plots. May break on other plotly plots..  
    Downsamples the points in each category of a violin plot to ensure an even number of points across all categories.

    Parameters:
        fig (plotly.graph_objects.Figure): The input violin plot figure.
        max_points (int, optional): The maximum number of points to downsample to. Defaults to None.
        max_points_pct (float, optional): The percentage of the smallest sample size to take. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: The modified violin plot figure with downsampled points.
    """
    # Calculate the minimum number of points among all categories if max_points or percentage is not provided
    if max_points is None and max_points_pct is None:
        min_points = min(len(trace.y) for trace in fig.data)
    elif max_points is None and max_points_pct is not None:
        smallest_sample = min(len(trace.y) for trace in fig.data)
        min_points = int(smallest_sample * max_points_pct)
    else:
        min_points = max_points

    # Iterate through each trace in fig.data
    for trace in fig.data:
        # Get the number of points in the current trace
        num_points = len(trace.y)
    
        # Downsample the points if the number of points is greater than the minimum
        if num_points > min_points:
            # Use numpy's linspace to evenly select downsampled indices
            if sample_method == 'random_sample': 
                random.seed(seed)
                downsampled_indices = random.sample(range(num_points), min_points)
            elif sample_method == 'linspace': 
                downsampled_indices = np.linspace(0, num_points - 1, min_points, dtype=int)
            
            # Update the trace with the downsampled points
            trace.y = [trace.y[i] for i in downsampled_indices]
            if trace.x is not None:
                trace.x = [trace.x[i] for i in downsampled_indices]
    
    return fig

def add_p_value_annotation(fig, 
                           array_columns, 
                           just_annotate = None,
                           test_type = 'ranksums', 
                           popmean = None, 
                           y_padding = True, 
                           subplot=None, 
                           include_tstat=None, 
                           p_round=3,
                           _format=dict(interline=0.07, text_height=1.07, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    
    assert test_type in ['ranksums', 'ttest_ind', 'ttest_rel', 'ttest_1samp'] , "Please specify test_type to be either ranksums or ttest"
    if test_type == 'ttest_1samp': 
        assert popmean is not None, "ttest_1samp requires popmean value"
    
    if just_annotate is not None: 
        assert len(just_annotate) == len(array_columns), "'just_annotate' and 'array_columns' len must be identical "
    
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    if y_padding:
        for i in range(len(array_columns)):
            y_range[i] = [1.01+i*_format['interline'], 1.02+i*_format['interline']]
    else: 
        for i in range(len(array_columns)):
            y_range[i] = [1.01+_format['interline'], 1.02+_format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str =str(subplot)
        indices = [] #Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            #print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        #print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        #print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        if test_type == 'ttest_ind': 
            tstat, pvalue = stats.ttest_ind(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
                equal_var=False,
            )
        elif test_type == 'ttest_rel':
            tstat, pvalue = stats.ttest_rel(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
            )
        elif test_type == 'ranksums':
            tstat, pvalue = stats.ranksums(
                fig_dict['data'][data_pair[0]]['y'],
                fig_dict['data'][data_pair[1]]['y'],
            )
        elif test_type == 'ttest_1samp':
            tstat, pvalue = stats.ttest_1samp(
                fig_dict['data'][data_pair[0]]['y'],
                popmean = popmean
            )
       
        if include_tstat: 
            symbol = format_pvalue(pvalue, p_round, t = tstat) 
        else: 
            symbol = format_pvalue(pvalue, p_round)
            
        if column_pair[0] != column_pair[1]: # If the column pair is the same, don't label lines
            # Vertical line
            fig.add_shape(type="line",
                xref="x"+subplot_str, yref="y"+subplot_str+" domain",
                x0=column_pair[0], y0=y_range[index][0], 
                x1=column_pair[0], y1=y_range[index][1],
                line=dict(color=_format['color'], width=2,)
            )
            # Horizontal line
            fig.add_shape(type="line",
                xref="x"+subplot_str, yref="y"+subplot_str+" domain",
                x0=column_pair[0], y0=y_range[index][1], 
                x1=column_pair[1], y1=y_range[index][1],
                line=dict(color=_format['color'], width=2,)
            )
            # Vertical line
            fig.add_shape(type="line",
                xref="x"+subplot_str, yref="y"+subplot_str+" domain",
                x0=column_pair[1], y0=y_range[index][0], 
                x1=column_pair[1], y1=y_range[index][1],
                line=dict(color=_format['color'], width=2,)
            )
        
        # If just_annotate (manual annotations) hard overwrites calculated stats in this function. Merely provides a 'symbol' to annotate
        if just_annotate is not None: 
            symbol = just_annotate[index]
        
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*_format['text_height'],
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))
    return fig

def format_pvalue(pvalue, p_round = 3, t = None):
    """
    Format a p-value as a string with significance symbols.

    Parameters:
    pvalue (float): The p-value to be formatted.

    Returns:
    str: The formatted p-value string with significance symbols.
    """
    if pvalue >= 0.05:
        symbol = f'ns <br>p={round(pvalue, p_round)}'
    elif pvalue >= 0.01:
        symbol = f'* <br>p={round(pvalue, p_round)}'
    elif pvalue >= 0.001:
        symbol = f'** <br>p={round(pvalue, p_round)}'
    else:
        symbol = f'*** <br>p={round(pvalue, p_round)}'
    
    if t is not None: 
        symbol += f'<br>t={round(t, 3)}'
        
    return symbol


from scipy.spatial import distance

def umap_euclidean_distance(umap_df, 
                            by,
                            coordinates = 'pca',
                            between='nostril', 
                            include_shuffle = True, 
                            shuffled_fraction = 0.3):
    assert coordinates in ['pca', 'umap'], print('Please specify coordinates to be either pca or umap')
    
    # Calculate pairwise distances
    distances = []

    groups = list(umap_df[by].sort_values().unique())
    if include_shuffle:
        groups += ['shuffled']
    
    for group in groups:
        if (group == 'shuffled'): 
            unique_top_Olfr = umap_df.sample(frac = shuffled_fraction)['top_Olfr'].unique()
        else: 
            unique_top_Olfr = umap_df[umap_df[by] == group]['top_Olfr'].unique()
        
        for olfr in unique_top_Olfr:
            olfr_data = umap_df[umap_df['top_Olfr'] == olfr]
            
            if coordinates == 'pca': 
                A_coords = olfr_data[olfr_data[between] == umap_df[between].unique()[0]][['pca_x', 'pca_y']].values
                B_coords = olfr_data[olfr_data[between] == umap_df[between].unique()[1]][['pca_x', 'pca_y']].values
            elif coordinates == 'umap':
                A_coords = olfr_data[olfr_data[between] == umap_df[between].unique()[0]][['umap_x', 'umap_y']].values
                B_coords = olfr_data[olfr_data[between] == umap_df[between].unique()[1]][['umap_x', 'umap_y']].values

            for A_point in A_coords:
                for B_point in B_coords:
                    dist = distance.euclidean(A_point, B_point)
                    distances.append({'top_Olfr': olfr, f'{coordinates}_distance': dist, 'group': group})

    # Create a DataFrame with pairwise distances
    return pd.DataFrame(distances)



def update_boxen(ax, cls="k", lw=1.5, ls="--"):
    """Update lines and points in seaborn boxenplot

    Parameters
    ----------
    ax : matplotlib.axes
        Axes containing a boxenplot
    """
    for a in ax.lines:
        a.set_color(cls)
        a.set_linewidth(lw)
        a.set_linestyle(ls)
        a.set_alpha(1)
    for a in ax.collections:
        if isinstance(a, mpl.collections.PatchCollection):
            # remove line surround each box
            a.set_linewidth(0)
        else:
            # remove outlier points
            a.set_alpha(0)
            
            

def distinct_colors(label_list, category=None, custom_color=None, random_state=0):
    """
    Generate distinct colors for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate distinct colors.
    category (str): Category of distinct colors. Options are 'warm', 'floral', 'rainbow', or None for random. Default is None.

    Returns:
    dict: A dictionary where labels are keys and distinct colors (in hexadecimal format) are values.

    Example:
    >>> labels = ['A', 'B', 'C']
    >>> color_mapping = distinct_colors(labels, category='warm')
    >>> print(color_mapping)
    {'A': '#fabebe', 'B': '#ffd8b1', 'C': '#fffac8'}
    """
    random.seed(random_state)
    
    warm_colors = ['#fabebe', '#ffd8b1', '#fffac8', '#ffe119', '#ff7f00', '#e6194B']
    floral_colors = ['#bfef45', '#fabed4', '#aaffc3', '#ffd8b1', '#dcbeff', '#a9a9a9']
    rainbow_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    pastel_colors = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', 
                     '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928', 
                     '#8DD3C7', '#BEBADA', '#FFED6F']
    
    color_dict = {}

    if custom_color is not None: 
        assert len(custom_color) >= len(label_list), "Provided label_list needs to be shorter than provided custom_color"
        for i, _label in enumerate(label_list): 
            color_dict[_label] = custom_color[i]
        return color_dict

    color_palette = None
    if category == 'warm':
        color_palette = random.sample(warm_colors, len(warm_colors))
    elif category == 'floral':
        color_palette = random.sample(floral_colors, len(floral_colors))
    elif category == 'rainbow':
        color_palette = random.sample(rainbow_colors, len(rainbow_colors))
    elif category == 'pastel': 
        color_palette = random.sample(pastel_colors, len(pastel_colors))
    else:
        color_palette = random.sample(warm_colors + floral_colors + rainbow_colors + pastel_colors, len(label_list))
    
    for i, label in enumerate(label_list):
        color_dict[label] = color_palette[i % len(color_palette)]
    
    return color_dict


import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def continuous_colors(label_list, colormap='viridis', custom_color=None, orders=None):
    """
    Generate continuous colors for a list of labels.

    Parameters:
    label_list (list): A list of labels for which you want to generate continuous colors.
    colormap (str or matplotlib colormap, optional): The colormap to use for color scaling. Default is 'viridis'.
    custom_color (list, optional): A list of color tuples defining the custom colormap.
                                   Default is None.
    orders (list, optional): A list defining the hierarchy of label_list. Default is None.

    Returns:
    dict: A dictionary where labels are keys and continuous colors (in hexadecimal format) are values.

    Example:
    >>> labels = ['A', 'B']
    >>> custom_color = [(0, '#DBE5EB'), (0.5, '#67879B'), (1, '#073763')]
    >>> color_mapping = continuous_colors(labels, custom_color=custom_color)
    >>> print(color_mapping)
    {'A': '#DBE5EB', 'B': '#073763'}
    """
    color_dict = {}

    # Choose colormap
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Generate custom colormap
    if custom_color is not None:
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_color)
    else:
        custom_cmap = None

    # Generate colors
    num_labels = len(label_list)
    for i, label in enumerate(label_list):
        if custom_cmap is not None:
            norm_color = i / (num_labels - 1)  # Normalize color index
            color = cm.colors.rgb2hex(custom_cmap(norm_color))
        else:
            color = cm.colors.rgb2hex(cmap(i / (num_labels - 1)))  # Normalize color index
        color_dict[label] = color

    # Reorder color_dict based on orders if provided
    if orders is not None:
        color_dict = {label: color_dict[label] for label in orders if label in color_dict}

    return color_dict


    
    
from scipy.stats import linregress


def label_ranksums_between_labels(fig, labels):
    """
    Computes the statistical test (Wilcoxon rank-sum test) between two groups of data based on the provided labels.
    
    Parameters:
    - fig (plotly.graph_objs._figure.Figure): The plotly figure object.
    - labels (list): A list containing two labels representing the groups for the statistical test.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): The updated plotly figure object with annotations for the statistical test results.
    """
    y_data = {}
    y_max = 0
    for _label in labels:
        y_data[_label] = [_d.y for _d in fig.data if _d.name == _label]       
        if y_max < np.max(y_data[_label]):
            y_max = np.max(y_data[_label])

    for i in range(len(y_data[_label][0])):
        y_list = []
        for _label in labels: 
            y_list.append([j[i] for j in y_data[_label]])
        _stat, _pval = stats.ranksums(y_list[0], y_list[1])
        fig.add_annotation(dict(font=dict(color='black',size=14),
                                x=i, y=y_max * 1.2,
                                showarrow=False,
                                text=format_pvalue(_pval)
                                ))
    return fig


from itertools import chain

def plot_olfr_lines(DE_Olfr_df, FDR_group, FDR_cutoff=0.05, 
                    plot_by='symbol',
                    manual_color={'KO+_Olfr': '#EF5350', 
                                  'KO-_Olfr': '#66BB6A'}, 
                    labels={'KO+_Olfr': 'greater_than', 
                            'KO-_Olfr': 'lesser_than'},
                    opacity_marker=0.2,
                    max_genes=np.inf,
                    exclude_group=None, 
                    trendline=False, 
                    std_shade=True, 
                    opactiy_shade=0.3,
                    plot_title = '', 
                    plot_xaxis_title = '', 
                    plot_yaxis_title = ''):
    
    """
    Plots Olfr lines and computes the average logFC and its variance across different groups.

    Parameters:
    - DE_Olfr_df (pandas.DataFrame): DataFrame containing Olfr data.
    - FDR_group (str): The group for which the plot is generated.
    - FDR_cutoff (float): The FDR cutoff value.
    - manual_color (dict): Dictionary mapping labels to color codes.
    - labels (dict): Dictionary mapping labels to group comparison types ('greater_than' or 'lesser_than').
    - exclude_group (str): The group to exclude from plotting.
    - plot_title (str): Title for the plot.
    - plot_xaxis_title (str): Title for the x-axis.
    - plot_yaxis_title (str): Title for the y-axis.

    Returns:
    - None
    """

    # subset = DE_Olfr_df[DE_Olfr_df.group != exclude_group] # Exclude data
    subset = DE_Olfr_df.copy()
    fig = go.Figure()
    # Subset for olfrs in different group categories 
    gene_dict = {}
    for _label in labels: 
        gene_dict[_label] = []
        if labels[_label] == 'greater_than': 
            temp_df = subset[(subset['group'].isin(FDR_group))]
            for _gene in temp_df[plot_by].unique(): 
                if len(gene_dict[_label]) < max_genes: 
                    if np.all((temp_df[(temp_df[plot_by] == _gene)].FDR < FDR_cutoff) & 
                              (temp_df[(temp_df[plot_by] == _gene)].logFC > 0)): gene_dict[_label].append(_gene) 
                else: 
                    print(f'{_label} max genes of {max_genes} reached.')
                    break
        elif labels[_label] == 'lesser_than': 
            temp_df = subset[(subset['group'].isin(FDR_group))]
            for _gene in temp_df[plot_by].unique(): 
                if len(gene_dict[_label]) < max_genes: 
                    if np.all((temp_df[(temp_df['symbol'] == _gene)].FDR < FDR_cutoff) & 
                              (temp_df[(temp_df[plot_by] == _gene)].logFC < 0)): gene_dict[_label].append(_gene) 
                else: 
                    print(f'{_label} max genes of {max_genes} reached.')
                    break
        else:
            return "ERROR: Labels dictionary has to contain \'greater_than\' or \'leser_than\'"
        # gene_list = list(chain.from_iterable(gene_dict.values()))
        gene_list = gene_dict[_label]
        subset_df = subset[subset[plot_by].isin(gene_list)]
        if exclude_group: 
            subset_df = subset_df[~subset_df.group.isin(exclude_group)].copy()
        
        plotlegend=True
        # Plot individual lines for each Olfr
        for _olfr in gene_list: 
            plot_df = subset_df[subset_df[plot_by] == _olfr].sort_values('group')
            fig.add_traces(go.Scatter(x = plot_df.group, 
                                        y = plot_df.logFC,
                                        name = _label, 
                                        mode = 'lines+markers', 
                                        hovertext = _olfr,
                                        legendgroup = _label, 
                                        opacity = opacity_marker,
                                        marker=dict(size = 10, 
                                                    color = manual_color[_label]), 
                                        showlegend=plotlegend
                                        ))
            plotlegend = False
        
        # Check if subset_df is empty, meaning no values values that met the FDR and criteria cutoff 
        if len(subset_df[plot_by].unique()) <= 1: 
            print(f'{_label} in {FDR_group} skipped')
            break
        
        # Add trendline trace
        # Calculate the average logFC for each group
        avg_logFC = subset_df.groupby('group')['logFC'].agg(['mean', 'std']).reset_index().dropna()
        if trendline: 
            slope, intercept = np.polyfit(range(len(avg_logFC)), avg_logFC['mean'], 1)
            trendline_logFC = slope * np.arange(len(avg_logFC)) + intercept
            fig.add_traces(go.Scatter(x=avg_logFC.group, 
                                        y=trendline_logFC, mode='lines', 
                                        line=dict(color = manual_color[_label], 
                                                width = 1)))
        if std_shade: 
        # Add shaded regions for standard deviation
            slope, intercept, _, _, _ = linregress(range(len(avg_logFC)), avg_logFC['mean'])
            upper_bound = avg_logFC['mean'] + avg_logFC['std']
            lower_bound = avg_logFC['mean'] - avg_logFC['std']
            fig.add_trace(go.Scatter(x=avg_logFC['group'].tolist() + avg_logFC['group'].tolist()[::-1],
                                        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                                        fill='toself',
                                        opacity = opactiy_shade,
                                        fillcolor=manual_color[_label],
                                        line=dict(color=manual_color[_label]),
                                        name=f'{_label} Variance'))

    # Stat test
    fig = label_ranksums_between_labels(fig, labels)

    fig.update_traces( 
        textposition='top center',
        # hovertemplate = '<b>%{hovertext}</b>'+'<br>logFC: %{y}<br>'
        )

    label_n = ', '.join([f'{key}: {len(gene_dict[key])}' for key in gene_dict])

    fig.update_layout(
        title = f'{plot_title}<br><sup>{FDR_group}, FDR < {FDR_cutoff}, {label_n}</sup>',
        xaxis=dict(title=plot_xaxis_title),
        yaxis=dict(title=plot_yaxis_title),
        template='simple_white'
    )
    return fig