import streamlit as st
import dataset
import modelo




def main():
    """Aplicação Visual"""
    st.title("Car Evaluation - Deep Learning ")
    st.subheader("Tópicos em Inteligencia Artificial (ELE0606)")


    #Menu
    menu = ["Análise do Dataset","Modelo"]
    opcoes  = st.sidebar.selectbox("Selecione uma das opções",menu)

    #importar dataframe:
    data = dataset.importar_dataframe()

    # Setando as abas
    match opcoes:
        case "Análise do Dataset":
            st.header("Análise do Dataset")

            dataset.data_set(data)
            
        
        case "Modelo":
            st.header("Modelo")
            st.write("Nessa aba vamos construir nosso modelo, treina-lo e, por fim, utiliza-lo para fazer predições com ele")
            modelo.modelo(data)
   

    
if __name__=='__main__':
    main()