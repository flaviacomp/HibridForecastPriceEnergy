import seaborn as sns
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft
import matplotlib.pyplot as plt
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams["axes.formatter.limits"] = (-99, 99)
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(1234)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.seasonal import STL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import Huber
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dateutil.relativedelta import relativedelta
import random
import os

plt.style.use('seaborn-v0_8')

def preparacao_dados(caminho_db, separador):

    ### CARREGANDO BANCO DE DADOS
    df = pd.read_csv(caminho_db, delimiter=separador, decimal=',')
    df.set_index("DATA", inplace=True)
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

    ### NORMALIZANDO OS DADOS
    df_ = df.copy()
    scaler_LPC = MinMaxScaler(feature_range=(1, 2))
    df_['LPC_SECO_NORMALIZADA'] = scaler_LPC.fit_transform(df_['LPC_SECO'].values.reshape(-1, 1))
    df_.index = pd.to_datetime(df_.index, format='%d/%m/%Y')

    ### DECOMPONDO A SÉRIE TEMPORAL
    stl = STL(df_['LPC_SECO_NORMALIZADA'], seasonal=53, period=52)
    result = stl.fit()
    df_['trend'] = result.trend
    df_['seasonal'] = result.seasonal
    df_['residual'] = result.resid

    return df_, scaler_LPC

def treinando_sazonalidade(df_normalizado, meses_forecast, start_forecast):
    SARIMA_model_seasonal = SARIMAX(df_normalizado['seasonal'].loc[df_normalizado.index <= meses_forecast[0]].dropna(),
                                    order=(2, 0, 1), seasonal_order=(1, 1, 1, 52), simple_differencing=False)
    SARIMA_model_fit_seasonal = SARIMA_model_seasonal.fit(disp=False)
    forecast_result_seasonal = SARIMA_model_fit_seasonal.get_forecast(steps=52 * 10)
    forecast_mean_seasonal = forecast_result_seasonal.predicted_mean
    confidence_intervals_seasonal = forecast_result_seasonal.conf_int()

    forecast_result_seasonal.predicted_mean = pd.Series(forecast_result_seasonal.predicted_mean.values,
                                                        index=meses_forecast)
    return forecast_result_seasonal.predicted_mean

def treinando_tendencia(df_normalizado, meses_forecast, treinar_novo_modelo):
    #### SARIMAX 3 ANOS
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_model_trend = SARIMAX(df_normalizado['trend'].loc[df_normalizado.index <= meses_forecast[0]].dropna(), order=(6, 0, 1),
                                 seasonal_order=(1, 0, 1, 52), simple_differencing=False)
    SARIMA_model_fit_trend = SARIMA_model_trend.fit(disp=True)
    forecast_result_trend = SARIMA_model_fit_trend.get_forecast(steps=52 * 10)

    forecast_result_trend.predicted_mean = pd.Series(forecast_result_trend.predicted_mean.values, index=meses_forecast)
    forecast_mean_trend = forecast_result_trend.predicted_mean
    confidence_intervals_trend = forecast_result_trend.conf_int()

    ### LSTM 7 ANOS
    # Definir a semente para reprodutibilidade
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Configuração adicional para garantir determinismo no TensorFlow
    tf.config.experimental.enable_op_determinism()

    # Preparar os dados para o modelo LSTM
    def create_dataset(series, time_step=1):
        data = series.values.reshape(-1, 1)
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    ## usando dois anos da projeção do sarimax como entrada do modelo
    print(meses_forecast[0], meses_forecast[0] + relativedelta(years=3))
    df_concat_sarimax = pd.concat([df_normalizado['trend'].loc[df_normalizado.index <= meses_forecast[0]],
                                   forecast_mean_trend.loc[forecast_mean_trend.index <= meses_forecast[0] + relativedelta(years=3)]])

    # Criar dataset
    time_step = 156  # Usando um ano de histórico para prever o próximo valor
    X, Y = create_dataset(df_concat_sarimax, time_step)

    # Redimensionar para LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Dividir em conjunto de treino e teste
    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

    # Redimensionar para LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    if(treinar_novo_modelo):
        # Construir o modelo LSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01))))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        # Compilar modelo
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())

        # Treinar com early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, Y_train, batch_size=32, epochs=300, validation_data=(X_test, Y_test), callbacks=[early_stopping])
        model.save('lstm_model.h5')
    else:
        model = load_model('lstm_model.h5')

    # Previsão e projeção da tendência
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Prever os próximos 10 anos (120 meses)
    future_steps = 363
    last_values = X_test[-1]

    lstm_forecast = []
    for _ in range(future_steps):
        prediction = model.predict(last_values.reshape(1, time_step, 1))
        lstm_forecast.append(prediction[0, 0])
        last_values = np.append(last_values[1:], prediction)

    # Avaliação do modelo,
    print("Treino:")
    print("mean_squared_error: ", mean_squared_error(Y_train, train_predict))
    print("mean_absolute_error: ", mean_absolute_error(Y_train, train_predict))
    print("Teste:")
    print("mean_squared_error: ", mean_squared_error(Y_test, test_predict))
    print("mean_absolute_error: ", mean_absolute_error(Y_test, test_predict))

    ### JUNTANDO AS PROJEÇÕES
    meses_lstm = pd.date_range(start=meses_forecast[0] + relativedelta(years=3), end=meses_forecast[-1] , freq='W')
    lstm_forecast = pd.Series(lstm_forecast, index=meses_lstm)
    df_concat_sarimax_lstm = pd.concat([forecast_mean_trend.loc[forecast_mean_trend.index <= meses_forecast[0] + relativedelta(years=3)], lstm_forecast])

    return df_concat_sarimax_lstm

def criando_modelo_xgboost(df_):
    df_normalizado = pd.DataFrame({'LPC_SECO_NORMALIZADA': df_["LPC_SECO_NORMALIZADA"],
                                   'seasonal': df_['seasonal'],
                                   'residual': df_['residual'],
                                   'volatility': df_['residual'].rolling(window=3).std(),
                                   'rolling_mean_5': df_['residual'].shift(1).rolling(window=5).mean(),
                                   'rolling_mean_12': df_['residual'].shift(1).rolling(window=12).mean(),
                                   }, index=df_.index)
    df_normalizado = df_normalizado.dropna()
    df_normalizado.index = pd.to_datetime(df_normalizado.index)
    # Aplicando o one-hot-vector
    df_normalizado['month'] = df_normalizado.index.month.astype(int)
    df_one_hot = pd.get_dummies(df_normalizado['month'], columns=['month'], prefix='month').astype(int)
    df_normalizado = df_normalizado.drop(columns=['month'])
    df_normalizado = pd.concat([df_normalizado, df_one_hot], axis=1)

    # criando lags features
    num_lags = 9

    for lag in range(1, num_lags + 1):
        df_normalizado[f'lag_{lag + 3}'] = df_normalizado['residual'].shift(lag + 3)
        df_normalizado[f'lag_{lag + 3}'] = df_normalizado[f'lag_{lag + 3}'].bfill()  # prestar atenção aqui

    # Splitting features and target
    X = df_normalizado.drop(columns=['residual', 'LPC_SECO_NORMALIZADA'], axis=1).loc[
        df_normalizado.index < "2023-12-31"]
    y = df_normalizado['residual'].loc[df_normalizado.index < "2023-12-31"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)

    # Definir a função de avaliação personalizada para MAPE
    def mape_eval(preds, dtrain):
        labels = dtrain.get_label()
        return 'mape', np.mean(np.abs((labels - preds) / (labels + 1))) * 100

    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Treinar o modelo
    eval_result = {}
    history = best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MSE:", mse)
    print("--------------------------------------")

    return best_model

def treinando_residuo(df_, forecast_result_seasonal, meses_forecast):
    best_model = criando_modelo_xgboost(df_)

    num_lags = 9
    # Função para fazer previsões iterativas
    def iterative_prediction(model, df, target, steps, data_inicial):
        # Cria uma cópia do dataframe para evitar modificar o original
        df_pred = df.copy()

        for step in range(steps):

            data_inicial = data_inicial + pd.DateOffset(weeks=1)

            # Convertendo o mês em one-hot encoding
            df_temp = pd.DataFrame({'month': [data_inicial.month]})
            df_one_hot = pd.get_dummies(df_temp, columns=['month'], prefix='month').astype(int)

            for i in range(1, 13):
                if f'month_{i}' not in df_one_hot.columns:
                    df_one_hot[f'month_{i}'] = 0

            new_data = {
                'seasonal': forecast_result_seasonal[step],
                'volatility': df_pred['residual'].rolling(window=3).std().iloc[-1] * 1.1,  # 1.1
                'rolling_mean_5': df_pred['residual'].rolling(window=5).mean().iloc[-1] * 1.6,  # 1.3
                'rolling_mean_12': df_pred['residual'].rolling(window=12).mean().iloc[-1] * 1.5,  # 1.5
                'month_1': df_one_hot['month_1'].values[0],
                'month_2': df_one_hot['month_2'].values[0],
                'month_3': df_one_hot['month_3'].values[0],
                'month_4': df_one_hot['month_4'].values[0],
                'month_5': df_one_hot['month_5'].values[0],
                'month_6': df_one_hot['month_6'].values[0],
                'month_7': df_one_hot['month_7'].values[0],
                'month_8': df_one_hot['month_8'].values[0],
                'month_9': df_one_hot['month_9'].values[0],
                'month_10': df_one_hot['month_10'].values[0],
                'month_11': df_one_hot['month_11'].values[0],
                'month_12': df_one_hot['month_12'].values[0],
            }

            new_df = pd.DataFrame(new_data, index=[0])

            for lag in range(1, num_lags + 1):
                new_df[f'lag_{lag + 3}'] = df_pred['residual'].iloc[-(lag + 3)]
                new_df[f'lag_{lag + 3}'] = new_df[f'lag_{lag + 3}'].bfill()

            new_pred = model.predict(new_df)

            new_row = {
                'residual': new_pred[0],
                'seasonal': new_data['seasonal'],
                'volatility': new_data['volatility'],
                'rolling_mean_5': new_data['rolling_mean_5'],
                'rolling_mean_12': new_data['rolling_mean_12'],
                'month_1': new_data['month_1'],
                'month_2': new_data['month_2'],
                'month_3': new_data['month_3'],
                'month_4': new_data['month_4'],
                'month_5': new_data['month_5'],
                'month_6': new_data['month_6'],
                'month_7': new_data['month_7'],
                'month_8': new_data['month_8'],
                'month_9': new_data['month_9'],
                'month_10': new_data['month_10'],
                'month_11': new_data['month_11'],
                'month_12': new_data['month_12'],
            }

            new_row = pd.DataFrame(new_row, index=[0])

            for lag in range(1, num_lags + 1):
                new_row[f'lag_{lag + 3}'] = new_df[f'lag_{lag + 3}']

            new_row.index = [data_inicial]

            df_pred = pd.concat([df_pred, new_row], ignore_index=False)

        return df_pred

    # Número de passos de previsão (ex. 4 semanas por mês)
    steps = 519
    data_inicial = meses_forecast[0]

    # Usando os dados mais recentes como ponto de partida para as previsões
    df_predictions = iterative_prediction(best_model, df_normalizado.loc[(df_normalizado.index <= meses_forecast[0])],
                                          "residual", steps, data_inicial)


    return df_predictions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    treinar_novo_modelo = True

    df_normalizado, scaler_LPC = preparacao_dados('db.csv', ';')

    start_forecast = df_normalizado.index[-1]
    end_forecast = start_forecast + relativedelta(weeks=519)
    print(start_forecast, end_forecast)

    start_validation = df_normalizado.index[-1] - relativedelta(weeks=51)
    end_validation = start_forecast
    print(start_validation, end_validation)
    meses_forecast = pd.date_range(start=start_forecast, end=end_forecast, freq='W')
    meses_validation = pd.date_range(start=start_validation, end=end_validation, freq='W')

    forecast_result_seasonal = treinando_sazonalidade(df_normalizado,meses_forecast, start_forecast)
    print(forecast_result_seasonal)


    forecast_result_trend = treinando_tendencia(df_normalizado, meses_forecast, treinar_novo_modelo)
    print(forecast_result_trend)

    df_predictions = treinando_residuo(df_normalizado, forecast_result_seasonal, meses_forecast)

    print(df_predictions["residual"].iloc[df_predictions["residual"].index >= start_forecast])

    meses_xgboost = pd.date_range(start='2024-01-07', end='2033-12-22', freq='W')
    meses_1ano = pd.date_range(start='2024-01-07', end='2024-12-22', freq='W')


    df_retro_10anos = pd.DataFrame()
    df_retro_10anos['pred_residual'] = pd.Series(df_predictions["residual"].iloc[df_predictions["residual"].index >= start_forecast].values,
        index=meses_forecast)
    print(df_retro_10anos['pred_residual'])
    df_retro_10anos['forecast_normalize'] = forecast_result_seasonal + forecast_result_trend + df_retro_10anos['pred_residual']
    df_retro_10anos['forecast'] = scaler_LPC.inverse_transform(df_retro_10anos['forecast_normalize'].values.reshape(-1, 1))

    df_resultado = pd.DataFrame()

    df_resultado['forecast'] = df_retro_10anos['forecast']
    # Adicionar a zona de erro de 5%
    # Calcular os limites de erro de 5%
    df_resultado['upper_error_5'] = df_resultado['forecast'] * 1.05  # Limite superior de 5% a mais
    df_resultado['lower_error_5'] = df_resultado['forecast'] * 0.95  # Limite inferior de 5% a menos

    df_resultado.to_csv("resultForecast.csv", sep=';', decimal=',')

