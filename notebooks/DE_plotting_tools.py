
import pandas as pd
import numpy as np 

import plotly.express as px 
import plotly.graph_objects as go

import umap
from sklearn.decomposition import PCA


def compare_vol_plot(DE_df_list: list, 
                     DE_df_name: list, 
                     fig_title = '', 
                     fig_dimension = [700,500]):
    
    assert len(DE_df_list) == len(DE_df_name), print("Please ensure DE_df_list and DE_df_name len matches")
    
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

    # Add a line for FDR = 0.05
    fig.add_shape(type='line', x0=-10, x1=10,
                  y0=-np.log10(0.05), y1=-np.log10(0.05),
                  line=dict(color='violet', width=3, dash='dash'))

    fig.update_traces( 
        textposition='top center',
        hovertemplate =
        '<b>%{text}</b>' + 
        '<br>LogFC: %{x}'+
        '<br>FDR: %{y}<br>')

    fig.update_layout(
        title=fig_title,
        autosize=True,
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
    