import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from dataset import importar_dataframe

def tratamentos_iniciais(data:pd.DataFrame,predicao:bool=False):
    # Codificar variáveis categóricas
    label_encoders = {}
    for column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    if predicao ==False:
        num_classes = len(label_encoders['class'].classes_)
        return data, num_classes


    return data

def criar_modelo(X_train, camadas, neuronios, dropout, num_classes):
    modelo = Sequential()
    modelo.add(Input(shape=(X_train.shape[1],)))

    # Adiciona as camadas conforme especificado pelo usuário
    for i in range(camadas):
        modelo.add(Dense(neuronios[i], activation='relu'))
        modelo.add(BatchNormalization())
        modelo.add(Dropout(dropout))

    # Camada de saída
    modelo.add(Dense(num_classes, activation='softmax'))

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

def treinar_e_testar_modelo(model, X_train, y_train, X_test, y_test):
    # Early stopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[early_stopping])

    # Avaliar o modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)
    st.markdown(f'- Test accuracy: {test_acc*100:.2f}%')
    st.write(f'- Test loss: {test_loss*100:.2f}%')

    # Mostrar resultado:
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Precisão de Treinamento ')
    ax.plot(history.history['val_accuracy'], label='Precisão de Validação ')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Acurácia')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

def input_car_data():
    # Coletar dados do usuário usando selectbox para opções específicas
    custo_compra = st.selectbox(
        'Custo de compra do carro',
        ['vhigh', 'high', 'med', 'low']
    )
    
    custo_manutencao = st.selectbox(
        'Custo de manutenção do carro',
        ['vhigh', 'high', 'med', 'low']
    )
    
    numero_portas = st.selectbox(
        'Número de portas do carro',
        ['2', '3', '4', '5more']
    )
    
    pessoas_acomodacao = st.selectbox(
        'Número de pessoas que o carro pode acomodar',
        ['2', '4', 'more']
    )
    
    tamanho_porta_malas = st.selectbox(
        'Tamanho do porta-malas',
        ['small', 'med', 'big']
    )
    
    nivel_seguranca = st.selectbox(
        'Nível de segurança do carro',
        ['low', 'med', 'high']
    )
    
    data = {
        "buying": [custo_compra],
        "maint": [custo_manutencao],
        "doors": [numero_portas],
        "persons": [pessoas_acomodacao],
        "lug_boot": [tamanho_porta_malas],
        "safety": [nivel_seguranca]
    }

    return  pd.DataFrame(data)

def fazer_previsao(dados_brutos,model,X_train):
    dados_processados = tratamentos_iniciais(dados_brutos,predicao=True)
    
    # Garantir que os dados de entrada tenham o mesmo formato que os dados de treinamento
    dados_processados = dados_processados.reindex(columns=X_train.columns, fill_value=0)

    # Fazer previsões
    previsoes = model.predict(dados_processados)
    
    # Obter a classe com a maior probabilidade
    classes_preditas = np.argmax(previsoes, axis=1)
    
    st.subheader("Resultado:")

    match classes_preditas[0]:
        case 1:
            st.write("unacc")
        case 2:
            st.write("acc")
        case 3:
            st.write("good")
        case 4:
            st.write("vgood")
        
    

def modelo(data:pd.DataFrame, test_size=0.2) -> None:
    ## Steando Modelo
    with st.expander('Construção do Modelo'):
        st.write("Até onde foi visto, nossa base estava ainda em formato de códigos, vamos transformar em escalas númericas ")
        data, num_classes = tratamentos_iniciais(data)
        st.dataframe(data.head())

        st.subheader('Parâmetros:')

        # Separar features e rótulos
        X = data.drop('class', axis=1)
        y = data['class']
       
        # Obtendo parâmetros do usuário
        test_size = (int(st.slider('Percentual de "Test Size" (%):', min_value=1, max_value=100, step=1)))/100
        camadas = int(st.slider('Quantidade de Camadas:', min_value=1, max_value=5, step=1))
        neuronios = []
        for n in range(camadas):
            neuronios.append(int(st.selectbox(f'Quantidade de neurônios na camada {n+1}:', [2,4,8,16,32,64,128])))
       
        
        dropout = int(st.slider('Percentual de "Dropout" (%):', min_value=0, max_value=100, step=1)) / 100

        # Codificar rótulos como one-hot
        y = to_categorical(y, num_classes=num_classes)

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        # Criar modelo
        model = criar_modelo(X_train, camadas, neuronios, dropout, num_classes)
    with st.expander('Treinamento'):
        st.subheader('Resultados')
        st.write('Modelo criado, agora vamos treinar e avaliar nosso modelo')
        
        # Treinar
        treinar_e_testar_modelo(model, X_train, y_train, X_test, y_test)

    with st.expander('Fazer previsões'):
        
        st.subheader('Usar o modelo:')
        st.write('Com modelo criado e treinado, agora vamos usa-lo para fazer previsões')
        
        data_bruto = input_car_data()

        fazer_previsao(data_bruto,model,X_train)
