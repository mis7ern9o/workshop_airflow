import requests
import os
import json
import pandas as pd
import ast
from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

@dag(
    dag_id='weather',
    tags=['workshop', 'datascientest', 'airflow'],
    schedule_interval=None,
    start_date=days_ago(0)
)

def weather_dag():

    @task_group(group_id='weather_get_data')
    @task 
    def set_variables():
        Variable.set(key='DATA_RAW_PATH', value='/app/raw_files')
        Variable.set(key='DATA_CLEAN_PATH', value='/app/clean_data')
        Variable.set(key='OPEN_WEATHER_API', value='ac929e0a4ea954a65769a23ded82409e')
        Variable.set(key='CITY_LIST', Value="['paris', 'london', 'washington']")
        Variable.set(key='OPEN_WEATHER_URL', Value='https://api.openweathermap.org/data/2.5/weather')
        Variable.set(key='MODEL_PICKLE', Value='./app/model.pckl')
        Variable.set(key='MODEL_BEST_PICKLE', Value='/app/clean_data/best_model.pickle')

    @task
    def get_openweather_data():
        city_list = ast.literal_eval(Variable.get('CITY_LIST'))
        openweather_api = Variable.get('OPEN_WEATHER_API')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        all_weather_data = {}

        for city in city_list:        
            response = requests.get(base_url, params=dict(q=city, APPID=openweather_api))
            
            if response.status_code == 200:
                weather_data = response.json()
                all_weather_data[city] = weather_data
                print(f"Successfully fetched data for {city}")
            else:
                print(f"Failed to fetch data for {city}. Status code: {response.status_code}")

        filename = f"{timestamp}.json"
        filepath = os.path.join(Variable.get('DATA_RAW_PATH'), filename)

        # Write the data to a JSON file
        with open(filepath, 'w') as f:
            json.dump(all_weather_data, f, indent=4)

    def transform_data_into_csv(n_files=None, filename='data.csv'):
        parent_folder = Variable.get('DATA_RAW_PATH')
        target_folder = Variable.get('DATA_CLEAN_PATH')
        files = sorted(os.listdir(parent_folder), reverse=True)
        if n_files:
            files = files[:n_files]

        dfs = []

        for f in files:
            with open(os.path.join(parent_folder, f), 'r') as file:
                data_temp = json.load(file)
            for data_city in data_temp:
                dfs.append(
                    {
                        'temperature': data_city['main']['temp'],
                        'city': data_city['name'],
                        'pression': data_city['main']['pressure'],
                        'date': f.split('.')[0]
                    }
                )

        df = pd.DataFrame(dfs)

        print('\n', df.head(10))

        df.to_csv(os.path.join(target_folder, filename), index=False)

    @task
    def dashboard_data():
        filename = 'data.csv'
        transform_data_into_csv(n_files=20, filename=filename)

    @task
    def training_data():
        filename = 'fulldata.csv'
        transform_data_into_csv(n_files=None, filename=filename)

    set_variables()>>get_openweather_data()>>dashboard_data()>>training_data()

    @task_group(group_id='weather_model_train')
    @task
    def compute_model_score(model, X, y):
        # computing cross val
        cross_validation = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring='neg_mean_squared_error')

        model_score = cross_validation.mean()

        return model_score

    @task
    def train_and_save_model(model, X, y):
        path_to_model = Variable.get('MODEL_PICKLE')
        # training the model
        model.fit(X, y)
        # saving model
        print(str(model), 'saved at ', path_to_model)
        dump(model, path_to_model)

    def prepare_data():
        path_to_data = os.path.join(Variable.get('DATA_CLEAN_PATH'), 'fulldata.csv')
        # reading data
        df = pd.read_csv(path_to_data)
        # ordering data according to city and date
        df = df.sort_values(['city', 'date'], ascending=True)

        dfs = []

        for c in df['city'].unique():
            df_temp = df[df['city'] == c]

            # creating target
            df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

            # creating features
            for i in range(1, 10):
                df_temp.loc[:, 'temp_m-{}'.format(i)
                            ] = df_temp['temperature'].shift(-i)

            # deleting null values
            df_temp = df_temp.dropna()

            dfs.append(df_temp)

        # concatenating datasets
        df_final = pd.concat(
            dfs,
            axis=0,
            ignore_index=False
        )

        # deleting date variable
        df_final = df_final.drop(['date'], axis=1)

        # creating dummies for city variable
        df_final = pd.get_dummies(df_final)

        features = df_final.drop(['target'], axis=1)
        target = df_final['target']

        return {
            'X': features,
            'y': target
        }

    @task
    def compute_model():

        X, y = prepare_data()

        score_lr = compute_model_score(LinearRegression(), X, y)
        score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
        return {
            'score_lr': score_lr,
            'score_dt': score_dt
        }
    
    @task
    def score_model():
        # using neg_mean_square_error
        if score_lr < score_dt:
            train_and_save_model(
                LinearRegression(),
                X,
                y,
                Variable.get('MODEL_BEST_PICKLE')
            )
        else:
            train_and_save_model(
                DecisionTreeRegressor(),
                X,
                y,
                Variable.get('MODEL_BEST_PICKLE')
            )

    