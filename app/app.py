import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
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
    html.Label('Którego modelu chcesz użyć?'),
    dcc.Dropdown(
        options=[
            {'label': 'Bayes', 'value': 'bayes'},
            {'label': 'Sieć Neuronowa', 'value': 'siec'}
        ],
        id="model",
    ),

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

    html.Label('Zniżka w %'),
    dcc.Dropdown(
        options=[{"label": offered_discount, "value": offered_discount} for offered_discount in
                 df.offered_discount.unique()],
        id='offered_discount',
    ),

    html.Label('Dzień Tygodnia'),
    dcc.Dropdown(
        options=[{"label": weekday, "value": weekday} for weekday in
                 df.weekday.unique()],
        id='weekday',
    ),

    html.Label('Pora Dnia'),
    dcc.Dropdown(
        options=[{"label": time_of_day, "value": time_of_day} for time_of_day in
                 df.time_of_day.unique()],
        id='time_of_day',
    ),

    html.Div(["Cena: ",
              dcc.Input(id='price', value='100.99', type='text')]),

    html.Button(id='button', n_clicks=0, children='Wyslij'),

    html.Div(id='my-output')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='button', component_property='n_clicks'),
    Input(component_id='city', component_property='value'),
    Input(component_id='delivery_company', component_property='value'),
    Input(component_id='offered_discount', component_property='value'),
    Input(component_id='time_of_day', component_property='value'),
    Input(component_id='weekday', component_property='value'),
    Input(component_id='price', component_property='value'),
    Input(component_id='model', component_property='value')

)
def update_output_div(n_clicks, city, delivery_company,
                      offered_discount, time_of_day, weekday, price, model):
    if n_clicks == 0:
        return dash.no_update
    else:
        if model == "bayes":
            f = open('../models/bayes_1.0.0.pickle', 'rb')
        else:
            f = open('../models/bayes_1.0.0.pickle', 'rb')  # TODO change model
        bayes = pickle.load(f)

        # dictionary of lists
        dict = {
            'offered_discount': [offered_discount],
            'price': [price]
        }
        dict.update(DataPreprocessing.findCity(city))
        dict.update(DataPreprocessing.findDeliveryCompany(delivery_company))
        dict.update(DataPreprocessing.findTimeOfDay(time_of_day))
        dict.update(DataPreprocessing.findWeekday(weekday))
        print(dict)

        df = pd.DataFrame(dict)

        y_pred = bayes.predict(df)
        f.close()

        return 'Przewidywana ilość dni: {}' \
            .format(y_pred[0] + 1)  #TODO dodanie jeden bo modele zaokrąglają w dół


if __name__ == '__main__':
    app.run_server(debug=True)
