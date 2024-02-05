import pandas as pd 
import json
from rich import print

# le arquivo com os MSE de todos os modelos (antes de combinar)
with open('src/mse.txt', 'r') as file:
    content = file.read()
    
models_mse = json.loads(content)
submercados = ['N', 'NE', 'S', 'SE']

def calcular_mse(dataframe, coluna_real, coluna_predita):
    return mean_squared_error(dataframe[coluna_real], dataframe[coluna_predita])

# le os modelos finais
final_models = {subm: pd.read_csv(f'D:/modelo-combinado/src/final_model_{subm}.csv') for subm in submercados}

