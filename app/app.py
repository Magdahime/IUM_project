import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from data_preprocessing.DataLoader import DataLoader
from data_preprocessing.DataPreprocessing import DataPreprocessing

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = DataLoader.load_data_from_path("../data/")
df = DataPreprocessing.prepareDate(data)

app.layout = html.Div(children=[
    html.H1(children='Zamówienie'),

    html.Div(children='''
            Wprowadź dane zamówienia aby otrzymać przewidywany czas dostawy.
        '''),

    html.Label('Miasto'),
    dcc.Dropdown(
        options=[{"label": city, "value": city} for city in df.city.unique()],
        id='city',
    ),

    html.Label('Firma dostawcza'),
    dcc.Dropdown(
        options=[{"label": delivery_company, "value": delivery_company} for delivery_company in
                 df.delivery_company.unique()],
        id='delivery_company',
    ),

    html.Div(id='my-output')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='city', component_property='value'),
    Input(component_id='delivery_company', component_property='value')
)
def update_output_div(city, delivery_company):
    return 'Output: {}, {}'.format(city, delivery_company)


if __name__ == '__main__':
    app.run_server(debug=True)
