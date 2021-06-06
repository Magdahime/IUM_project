import os
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


def createPickles():
    if not os.path.exists('../UserPredictionData/bayes.pickle'):
        df = pd.DataFrame()
        f = open('../UserPredictionData/bayes.pickle', 'wb+')
        pickle.dump(df, f)
        f.close()

    if not os.path.exists('../UserPredictionData/neural.pickle'):
        df = pd.DataFrame()
        f = open('../UserPredictionData/neural.pickle', 'wb+')
        pickle.dump(df, f)
        f.close()


createPickles()

app.layout = html.Div(children=[
    html.H1(children='Zamówienie'),

    html.Div(children='''
            Wprowadź dane zamówienia aby otrzymać przewidywany czas dostawy.
        '''),

    html.Div(["Id użytkownika: ",
              dcc.Input(id='user_id', type='number')]),

    html.Label('Miasto'),
    dcc.Dropdown(
        options=[{"label": city, "value": city} for city in df.city.unique()],
        id='city',
        value="Konin",
    ),

    html.Label('Firma dostawcza'),
    dcc.Dropdown(
        options=[{"label": delivery_company, "value": delivery_company} for delivery_company in
                 df.delivery_company.unique()],
        id='delivery_company',
        value=620
    ),

    html.Label('Dzień Tygodnia'),
    dcc.Dropdown(
        options=[{"label": weekday, "value": weekday} for weekday in
                 df.weekday.unique()],
        id='weekday',
        value="Friday",
    ),

    html.Label('Pora Dnia'),
    dcc.Dropdown(
        options=[{"label": time_of_day, "value": time_of_day} for time_of_day in
                 df.time_of_day.unique()],
        id='time_of_day',
        value="Night",
    ),

    html.Label('Zniżka w %'),
    dcc.Dropdown(
        options=[{"label": offered_discount, "value": offered_discount} for offered_discount in
                 df.offered_discount.unique()],
        id='offered_discount',
        value=0
    ),

    html.Div(["Cena: ",
              dcc.Input(id='price', value='100.99', type='text')]),

    html.Button(id='button', n_clicks=0, children='Wyslij'),

    html.Div(id='my-output')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='button', component_property='n_clicks'),
    Input(component_id='user_id', component_property='value'),
    Input(component_id='city', component_property='value'),
    Input(component_id='delivery_company', component_property='value'),
    Input(component_id='time_of_day', component_property='value'),
    Input(component_id='weekday', component_property='value'),
    Input(component_id='offered_discount', component_property='value'),
    Input(component_id='price', component_property='value'),

)
def update_output_div(n_clicks, user_id, city, delivery_company,
                      time_of_day, weekday, offered_discount, price):
    if n_clicks == 0:
        return dash.no_update
    else:
        one_hot_encoded_data = {}
        modelname = hash_function(user_id)
        if modelname == "bayes":
            modelPickle = open('../models/bayes_1.0.0.pickle', 'rb')
            dataPickle = open('../UserPredictionData/bayes.pickle', 'rb')
            one_hot_encoded_data = {
                'offered_discount': [offered_discount],
                'price': [price]
            }
        elif modelname == "neural_net":
            modelPickle = open('../models/neural_n_1.0.0.pickle', 'rb')
            dataPickle = open('../UserPredictionData/neural.pickle', 'rb')
        model = pickle.load(modelPickle)
        oldPredictions = pickle.load(dataPickle)

        one_hot_encoded_data.update(DataPreprocessing.findCity(city))
        one_hot_encoded_data.update(
            DataPreprocessing.findDeliveryCompany(delivery_company))
        one_hot_encoded_data.update(
            DataPreprocessing.findTimeOfDay(time_of_day))
        one_hot_encoded_data.update(DataPreprocessing.findWeekday(weekday))

        y_pred = model.predict(pd.DataFrame(one_hot_encoded_data, index=[0]))

        oldPredictions = oldPredictions.append(
            add_id_and_predicted(one_hot_encoded_data, user_id, y_pred),
            ignore_index=True)

        if modelname == "bayes":
            save_pickle(oldPredictions, '../UserPredictionData/bayes.pickle')
        elif modelname == "neural_net":
            save_pickle(oldPredictions, '../UserPredictionData/neural.pickle')

        dataPickle.close()
        modelPickle.close()

        return 'Przewidywana ilość dni: {}' \
            .format(y_pred[0])


def save_pickle(df, path):
    fa = open(path, 'wb')
    pickle.dump(df, fa)
    fa.close()


def add_id_and_predicted(dictionary_data, user_id, y_pred):
    rememberData = {"id": user_id, "prediction": y_pred}
    rememberData.update(dictionary_data)
    rememberData_df = pd.DataFrame(rememberData, index=[0])
    return rememberData_df


def hash_function(user_id):
    if(hash(user_id) % 2 == 0):
        return "bayes"
    return "neural_net"


if __name__ == '__main__':
    app.run_server(debug=True)
