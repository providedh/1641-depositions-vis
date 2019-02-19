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

def date_string_to_date(date_string):
    return date(
            int(date_string.split('-')[0]), 
            int(date_string.split('-')[1]), 
            int(date_string.split('-')[2]))

def initialize():
    df = pd.read_json('./data/all_depositions.json')
    n_counties = len(geojson['features'])

    #Dates
    df['creation_date_period'] = pd.PeriodIndex(df['creation_date'], freq='d')

    df['creation_date_parsed'] = df['creation_date'].map(
        lambda x: date_string_to_date(x) if isinstance(x, str) else x)

    df['creation_year'] = df['creation_date'].str.slice(start=0, stop=4)
    df['creation_year'] = df['creation_year'].fillna('Unknown')

    #Places
    df.deponent_county = df.deponent_county.fillna('Unknown')

    return df    

df = initialize()


def create_timeline(dff):
    #Counts by Date
    dates_counts = dff['creation_date_parsed'].value_counts()
    dates_counts.sort_index(inplace=True)
    dates = dates_counts.index.tolist()
    dates_c = [dates_counts[d] for d in dates]

    return  {
                'data': [go.Histogram(
                    x=dates,
                    y=dates_c,
                    opacity = 0.8,
                    marker=dict(
                        color='#16632d'),
                    xbins=dict(
                        start=min(dates),
                        end=max(dates),
                        size='M1'
                        ),
                    autobinx=False)],
                'layout': go.Layout(
                    title='Timeline of depositions',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1,
                                     label='1m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=6,
                                     label='6m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=1,
                                    label='1y',
                                    step='year',
                                    stepmode='backward'),
                                dict(step='all')
                            ])
                        ),
                        rangeslider=dict(
                            visible = True
                        ),
                        type='date'
                    ),
                    width=800,
                    height=400)
            }
            

def create_map(dff):
    def scalarmappable(cmap, cmin, cmax):
        colormap = cm.get_cmap(cmap)
        norm = Normalize(vmin=cmin, vmax=cmax)
        return cm.ScalarMappable(norm=norm, cmap=colormap)

    def get_scatter_colors(sm, df, counties):
        grey = 'rgba(128,128,128,1)'
        return ['rgba' + str(sm.to_rgba(df[c], bytes = True, alpha = 1)) \
            if df[c] != 0 else grey for c in counties]

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
    county_names = sorted(list(set([k['properties']['CountyName'] for k in geojson['features']])))
    counts = dff['deponent_county'].value_counts()
    
    if 'Unknown' in counts.index:
        counts.drop('Unknown', inplace=True)

    for c in county_names:
        if c not in counts.index:
            counts[c] = 0
    
    counts.sort_index(inplace=True)

    # print(counts)

    colormap = 'Greens'
    cmin = counts.min()
    cmax = counts.max()

    lons, lats = get_centers(features)

    sm = scalarmappable(colormap, cmin, cmax)
    scatter_colors = get_scatter_colors(sm, counts, county_names)

    colorscale = get_colorscale(sm, counts, cmin, cmax)
    hover_text = get_hover_text(counts)


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

    return {
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
                    customdata= counts.index,
                    showlegend=False,
                    hoverinfo='text'
                )],
                'layout': go.Layout(
                    autosize=False,
                    width = 350,
                    height = 400,
                    hovermode='closest',
                    margin = dict(l=0, r=0, b=0, t=0),
                    mapbox=dict(
                        layers=layers,
                        accesstoken=mapbox_access_token,
                        bearing=0,
                        center=dict(
                            lat=53.5,
                            lon=-8
                        ),
                        pitch=0,
                        zoom=5.1,
                        style = 'light'
                    ),

                ) 
            }
            


app.layout = html.Div(children=[
    html.H1(children='1641 Depositions'),

    html.Div(id='row', className='row', children=[

        html.Div(dcc.Graph(
                        id='depositions-timeline'),
            className='six columns'),

        

        html.Div(dcc.Graph(
                        id='depositions-map'), 
            className='six columns')
    ]),

    html.Div(id='debug-div')
])

@app.callback(
    Output('depositions-map', 'figure'),
    [Input('depositions-timeline', 'relayoutData')])

def update_map(relayoutData):
    print(relayoutData)
    if relayoutData and ('xaxis.range[0]' in relayoutData or 'xaxis.range' in relayoutData):
        # print(selectedData['points'][0]['x'])
        if 'xaxis.range[0]' in relayoutData:
            start_date = relayoutData['xaxis.range[0]'].split(" ")[0]
            end_date = relayoutData['xaxis.range[1]'].split(" ")[0]
        else:
            start_date = relayoutData['xaxis.range'][0].split(" ")[0]
            end_date = relayoutData['xaxis.range'][1].split(" ")[0]
        print(start_date, end_date)
        dff = df[(df['creation_date_parsed'] >= date_string_to_date(start_date)) & \
        (df['creation_date_parsed'] <= date_string_to_date(end_date))]
        # print(dff)
        return create_map(dff)
    else:
        return create_map(df)

@app.callback(
    Output('depositions-timeline', 'figure'),
    [Input('depositions-map', 'clickData')])

def update_timeline(clickData):
    return create_timeline(df)


@app.callback(
        Output('debug-div', 'children'),
        [Input('depositions-timeline', 'relayoutData'),
        Input('depositions-map', 'clickData')])

def update_output_div(clickDataTimeline, clickDataMap):
    return 'You\'ve entered "{} and {}"'.format(clickDataTimeline, clickDataMap)
    # if clickData is not None:
    #     print(clickData)
    #     county_idx = clickData['points'][0]['pointIndex']
    #     # print(counties_counts[county_idx])
    #     return 'You\'ve entered "{}"'.format(clickData['points'][0]['text'])
    # else:
    #     return 'You\'ve entered None'


if __name__ == '__main__':
    app.run_server(debug=True)