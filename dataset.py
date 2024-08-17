import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
  
def importar_dataframe():
    # URL do arquivo CSV do conjunto de dados "Car Evaluation"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

    # Nomes das colunas do conjunto de dados
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

    # Carregar o dataset em um DataFrame
    df = pd.read_csv(url, header=None, names=columns)
    return df

def tabela_atributos():
    attributes = {
    "Atributo": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
    "Tipo": ["Categórico"]  * 6 + ["Rótulo"],
    "Descrição": [
        "Custo de compra do carro",
        "Custo de manutenção do carro",
        "Número de portas do carro",
        "Número de pessoas que o carro pode acomodar",
        "Tamanho do porta-malas",
        "Nível de segurança do carro",
        "Classificação geral do carro"
    ],
    "Categorias": [
        "vhigh, high, med, low",
        "vhigh, high, med, low",
        "2, 3, 4, 5more",
        "2, 4, more",
        "small, med, big",
        "low, med, high",
        "unacc, acc, good, vgood"
    ]
}

    # Criar um DataFrame para a tabela de atributos
    return pd.DataFrame(attributes)

def heat_map(df:pd.DataFrame):
    # Convertendo colunas categóricas para numéricas para calcular a correlação
    df_encoded = df.apply(lambda col: pd.factorize(col, sort=True)[0])

    # Calculando a matriz de correlação
    corr_matrix = df_encoded.corr()

    # Plotando o heatmap da matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=0.5)

    
      # Exibir o heatmap no Streamlit
    st.pyplot(plt)

def data_set(data:pd.DataFrame)->None:
     
        
        st.write(f"O nosso dataset estudo possui {data.shape[0]} linhas e {data.shape[1]} colunas, conforme na visão abaixo:")
        
        st.dataframe(data.head(10))

        st.write("Dessas colunas, temos 2 com atributos numéricos e 4 de atributos textuais. Cada qua com sua categoria. Além disso, uma última coluna de rótulo, chamada 'Class':")
        st.dataframe(tabela_atributos())
        st.markdown("Isso posto, vamos analisar sumariamente o nosso dataset por meio do método `describe()` do pandas.")
        st.write(data.describe())
        
        st.markdown("O que já pode nos revelar algumas têndencias, como os termos mais frequentes de cada coluna e a sua frequência. Por fim, para entender melhor ainda nossa distribuição de rotulos, vamos visulizar sua distribuição por meio de um _gráfico de setor_: ")
        
        fig, ax = plt.subplots(figsize=(5, 5))
        data['class'].value_counts().plot.pie(ax=ax)
        st.pyplot(fig)

        st.markdown("Para finalizar nossa análise de dados, vamos contruir um `heatmap`. A ideia é visualizar a correlação entre as variáveis do nosso dataframe ")
        heat_map(data)


if __name__=='__main__':
   df =  importar_dataframe()
   print(df.shape[0])