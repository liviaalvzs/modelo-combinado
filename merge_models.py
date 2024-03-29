import pandas as pd 
import json
from sklearn.metrics import mean_squared_error

# le arquivo com os MSE de todos os modelos (antes de combinar)
with open('src/mse.txt', 'r') as file:
    content = file.read()
    
models_mse = json.loads(content)
submercados = ['N', 'NE', 'S', 'SE']

def calcular_mse(dataframe, coluna_real, coluna_predita):
    return mean_squared_error(dataframe[coluna_real], dataframe[coluna_predita])

# le os modelos finais
final_models = {subm: pd.read_csv(f'src/final_model{subm}.csv') for subm in submercados}

new_models = {subm: pd.read_csv(f'src/modelo_estendido/final_model{subm}.csv') for subm in submercados}
