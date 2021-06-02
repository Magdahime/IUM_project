import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from Regresion.LinearRegresion import unite_sets
from data_preprocessing.DataLoader import DataLoader

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

d = DataLoader.load_data_from_path("../newData")

products = d.products
deliveries = d.deliveries
sessions = d.sessions
users = d.users

df = unite_sets(deliveries, products, sessions, users)

app.layout = html.Div(children=[
    html.H1(children='Zamówienie'),

    html.Div(children='''
            Wprowadź dane zamówienia aby otrzymać przewidywany czas dostawy.
        '''),

    html.Label('Miasto'),
    dcc.Dropdown(
        options=[{"label": city, "value": city} for city in df.city.unique()],
        id='miasto',
    ),

    html.Div(id='my-output')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='miasto', component_property='value')
)
def update_output_div(miasto):
    return 'Output: {}'.format(miasto)


if __name__ == '__main__':
    app.run_server(debug=True)
