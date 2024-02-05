import streamlit as st
import pandas as pd
import plotly.express as px
from merge_models import final_models, models_mse, calcular_mse

# Mapeamento de Regioes
regioes = {'N': 'Norte',
           'SE': 'Sudeste/Centro Oeste', 
           'S': 'Sul',
           'NE': 'Nordeste'} 

# Mapeamento de chaves
mapeamento_chaves = {
                    'rn500x2_sp': 'Redes Neurais - S/ Patamares (Relu 500Nx2)',
                     'rn500x2_cp': 'Redes Neurais - C/ Patamares (Relu 500Nx2)',
                     'rn500x2_sp_identity': 'Redes Neurais - S/ Patamares (Identity 500Nx2)',
                     'rn500x2_cp_identity': 'Redes Neurais - C/ Patamares (Identity 500Nx2)',
                     'rn500x3_sp_identity': 'Redes Neurais - S/ Patamares (Identity, 500Nx3)',
                     'rn500x3_cp_identity': 'Redes Neurais - C/ Patamares (Identity, 500Nx3)',
                     'rn1000x2_sp_identity': 'Redes Neurais - S/ Patamares (Identity, 1000N)',
                     'rn1000x2_cp_identity': 'Redes Neurais - C/ Patamares (Identity, 1000N)',
                     'quantile_sp': 'Regressao Quantilica - S/ Patamares',
                     'quantile_cp': 'Regressao Quantilica - C/ Patamares',
                     'svm_sp': 'SVM S/ Patamares',
                     'svm_cp': 'SVM C/ Patamares',
                     'prevcarga d': 'Prevcarga D',
                     'prevcarga d-1': 'Prevcarga D-1',
                     'prevcarga d-2': 'Prevcarga D-2',
                     'Modelo Combinado Safira': 'Modelo Combinado Safira',
                     }

# Função para criar o gráfico interativo
def create_plot(df, region, show_prevcarga):
    df['data_previsao'] = pd.to_datetime(df['data_previsao'])
    
    # Adiciona a coluna "Prevcarga" ao gráfico se a opção estiver selecionada
    y_columns = ['Carga', 'Modelo Combinado Safira']
    if show_prevcarga:
        y_columns.append('PrevCargaDessem')
    
    fig = px.line(df, x='data_previsao', y=y_columns,
                  title=f'Modelo Previsao de Carga - {regioes[region]}',
                  labels={'value': 'Carga'},
                  line_shape='linear', color_discrete_sequence=['#80c423', '#491a74', '#f35b04' if show_prevcarga else 'rgba(0,0,0,0)'])

    fig.add_trace(px.scatter(df, x='data_previsao', y='Carga', color_discrete_sequence=['#80c423']).data[0])
    fig.add_trace(px.scatter(df, x='data_previsao', y='Modelo Combinado Safira', color_discrete_sequence=['#491a74']).data[0])

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 day", step="day", stepmode="backward"),
                    dict(count=7, label="1 week", step="day", stepmode="backward"),
                    dict(count=14, label="2 weeks", step="day", stepmode="backward"),
                    dict(count=1, label="1 month", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

# Função para criar tabela
def create_table(valores, mapeamento_chaves):
    # Ordena os valores do dicionário pelo seu valor
    valores_ordenados = sorted(valores.items(), key=lambda x: x[1])
    
    # Mapeia as chaves conforme o dicionário de mapeamento
    valores_mapeados = [(mapeamento_chaves[chave], valor) for chave, valor in valores_ordenados]
    
    # Cria um DataFrame para a tabela
    df = pd.DataFrame(valores_mapeados, columns=['Modelo', 'Valor do MSE'])
    
    return df

def percentual_melhoria(mse_modelo1, mse_modelo2):
    return ((mse_modelo1 - mse_modelo2) / mse_modelo1) * 100


# Configuração da página para wide mode
st.set_page_config(layout="wide")

# Interface Streamlit
st.title('Modelo Combinado de Previsão de Carga')

# Seleção de região
region = st.selectbox('Selecione a região:', list(regioes.keys()))

# Adiciona a caixa de seleção "Exibir PrevCarga"
show_prevcarga = st.checkbox('Exibir PrevCarga')

df = final_models[region]
mse_values = models_mse[region]

df = df.rename(columns={'carga_prevista_final': 'Modelo Combinado Safira', 
                        'carga_real': 'Carga', 
                        'prevcarga': 'PrevCargaDessem'})

# Mostrar gráfico para a região selecionada
fig = create_plot(df, region, show_prevcarga)

# Cria a tabela
mse_modelo_combinado = calcular_mse(df, 'Carga', 'Modelo Combinado Safira')
min_mse_modelo, min_mse_valor = min(mse_values.items(), key=lambda x: x[1])

mse_values_copy = mse_values.copy()
mse_values_copy['Modelo Combinado Safira'] = mse_modelo_combinado
df_tabela = create_table(mse_values_copy, mapeamento_chaves)

# plota grafico
fig.update_layout(
    autosize=False, 
    height=800  
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### **MSE dos Modelos Individuais, Modelo Combinado e PrevCarga DESSEM**")

valor_mse_prevcarga_d = df_tabela.loc[df_tabela['Modelo'] == 'Prevcarga D', 'Valor do MSE'].values[0]
valor_mse_melhor_modelo = df_tabela['Valor do MSE'].min()

comp_safira_prevcarga = percentual_melhoria(valor_mse_prevcarga_d, mse_modelo_combinado)
comp_safira_melhor_modelo = percentual_melhoria(min_mse_valor, mse_modelo_combinado)

st.table(df_tabela)

st.markdown(f'### Percentual de melhoria entre...')
st.markdown(f'#### Modelo Combinado e o Melhor modelo ({min_mse_modelo}): **{comp_safira_melhor_modelo:.2f}%**')
st.markdown(f'#### Modelo Combinado e o PREVCARGA DESSEM: **{comp_safira_prevcarga:.2f}%**')
