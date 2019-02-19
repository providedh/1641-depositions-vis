# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import pandas as pd
import json

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

import os

from datetime import date



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


if os.environ['MAPBOX_ACCESS_TOKEN'] is None:
    sys.exit('Please provide a mapbox access token as environment variable: export MAPBOX_ACCESS_TOKEN=<Your token>')

mapbox_access_token = os.environ['MAPBOX_ACCESS_TOKEN']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open('./data/ireland-geo.json') as f:
    geojson = json.load(f)

def initialize():
    df = pd.read_json('./data/all_depositions.json')
    n_counties = len(geojson['features'])

    #Dates
    df['creation_date_period'] = pd.PeriodIndex(df['creation_date'], freq='d')

    df['creation_date_parsed'] = df['creation_date'].map(
        lambda x: date(
            int(x.split('-')[0]), 
            int(x.split('-')[1]), 
            int(x.split('-')[2])) if isinstance(x, str) else x)

    df['creation_year'] = df['creation_date'].str.slice(start=0, stop=4)
    df['creation_year'] = df['creation_year'].fillna('Unknown')

    #Places
    df.deponent_county = df.deponent_county.fillna('Unknown')

    return df    

df = initialize()


def create_timeline():
    #Counts by Date
    dates_counts = df['creation_date_parsed'].value_counts()
    dates_counts.sort_index(inplace=True)
    dates = dates_counts.index.tolist()
    dates_c = [dates_counts[d] for d in dates]

    return dcc.Graph(
                id='depositions-timeline',
                figure={
                    'data': [go.Scatter(
                        x=dates,
                        y=dates_c,
                        opacity = 0.8)],
                    'layout': go.Layout(
                        autosize=False,
                        width=800,
                        height=400)
                }
            )

def create_map():
    def scalarmappable(cmap, cmin, cmax):
        colormap = cm.get_cmap(cmap)
        norm = Normalize(vmin=cmin, vmax=cmax)
        return cm.ScalarMappable(norm=norm, cmap=colormap)

    def get_scatter_colors(sm, df):
        grey = 'rgba(128,128,128,1)'
        return ['rgba' + str(sm.to_rgba(m, bytes = True, alpha = 1)) if not np.isnan(m) else grey for m in df]

    def get_colorscale(sm, df, cmin, cmax):
        xrange = np.linspace(0, 1, len(df))
        values = np.linspace(cmin, cmax, len(df))

        return [[i, 'rgba' + str(sm.to_rgba(v, bytes = True))] for i,v in zip(xrange, values) ]

    def get_hover_text(counts) :
        text_value = (counts).round(2).astype(str)
        with_data = '<b>{}:</b> <br> {} depositions'
        return [with_data.format(p,v) for p,v in zip(counts.index, text_value)]

    def get_centers(features):
        lon, lat = [], []
        n_counties = len(features)
        for k in range(n_counties):
            geometry = features[k]['geometry']

            if geometry['type'] == 'Polygon':
                coords=np.array(geometry['coordinates'][0])
            elif geometry['type'] == 'MultiPolygon':
                longest_idx = -1
                max_length = -1
                for i in range(len(geometry['coordinates'])):
                    poly_length = len(geometry['coordinates'][i][0])
                    if max_length < poly_length:
                        max_length = poly_length
                        longest_idx = i
                coords=np.array(geometry['coordinates'][longest_idx][0])

            lon.append(sum(coords[:,0]) / len(coords[:,0]))
            lat.append(sum(coords[:,1]) / len(coords[:,1]))
                
        return lon, lat


    #Counts by County
    features = sorted(geojson['features'], key=lambda k: k['properties']['CountyName'])
    counts = df['deponent_county'].value_counts()
    counties_counts = counts.drop('Unknown')
    counties_counts.sort_index(inplace=True)

    colormap = 'Greens'
    cmin = counties_counts.min()
    cmax = counties_counts.max()

    lons, lats = get_centers(features)

    sm = scalarmappable(colormap, cmin, cmax)
    scatter_colors = get_scatter_colors(sm, counties_counts)
    colorscale = get_colorscale(sm, counties_counts, cmin, cmax)
    hover_text = get_hover_text(counties_counts)


    layers = ([dict(sourcetype = 'geojson',
                  source = features[k],
                  below = "",
                  type = 'line',    # the borders
                  line = dict(width = 1),
                  color = 'black',
                  ) for k in range(len(features))] +

            [dict(sourcetype = 'geojson',
                 source = features[k],
                 below="water",
                 type = 'fill',
                 color = scatter_colors[k],
                 opacity=1
                ) for k in range(len(features))]
            )

    return dcc.Graph(
                id='depositions-map',
                figure={
                    'data': [go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=scatter_colors,
                            showscale=True,
                            cmin=cmin,
                            cmax=cmax,
                            colorscale=colorscale,
                            colorbar=dict(tickformat = ".0")
                        ),
                        text=hover_text,
                        showlegend=False,
                        hoverinfo='text'
                    )],
                    'layout': go.Layout(
                        autosize=False,
                        width = 700,
                        height = 800,
                        hovermode='closest',
                        mapbox=dict(
                            layers=layers,
                            accesstoken=mapbox_access_token,
                            bearing=0,
                            center=dict(
                                lat=53.5,
                                lon=-8
                            ),
                            pitch=0,
                            zoom=6,
                            style = 'light'
                        ),
                    ) 
                }
            )



app.layout = html.Div(children=[
    html.H1(children='1641 Depositions'),

    html.Div(id='row', className='row', children=[

        html.Div([create_timeline()
            ], 
            className='six columns'),

        

        html.Div([
            create_map()
            ], 
            className='six columns')
    ]),

    html.Div(id='my-div')
])

@app.callback(
        Output('my-div', 'children'),
        [Input('depositions-map', 'clickData')])

def update_output_div(input_value):
    if input_value is not None:
        print(input_value)
        county_idx = input_value['points'][0]['pointIndex']
        # print(counties_counts[county_idx])
        return 'You\'ve entered "{}"'.format(input_value['points'][0]['text'])
    else:
        return 'You\'ve entered None'


if __name__ == '__main__':
    app.run_server(debug=True)