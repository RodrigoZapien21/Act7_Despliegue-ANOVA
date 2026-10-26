#Creamos el archivo de la APP en el interprete principal (Phyton)
#####################################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################
#Definimos la instancia
@st.cache_resource
######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_excel("salidas_almacenes_sin_outliers.xlsx")
    #Rellenamos nulos
   df =df.bfill()
   df =df.ffill()  
   #Creamos la lista de las variables categóricas sobre las que se generará la extracción de características(análisis univariado)
   Lista=['anio', 'mes', 'articulo_id', 'marca','salidas']
   return df, Lista
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, Lista = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("Salidas almacen")

#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.sidebar.selectbox(label= "Tipo de Análisis", options= ["Extracción de Características", 
                                              "Regresión Lineal", "Regresión No Lineal",
                                              "Regresión Logística", "ANOVA"])

# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box
    Variable_Cat= st.sidebar.selectbox(label= "Variables", options= Lista)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    #Ajustamos los nombre de las cabeceras de las columnas
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']
    
    #Generamos los encabezados para el dashboard
    st.title("Extracción de Características")
    

 
  ####################################################################################
 
 # CONTENIDO DE LA VISTA 5
if View == "ANOVA":
    #ANOVA 
    #Generamos la lista de variables numéricas
    numeric_df = df.select_dtypes(['float','int'])  #Devuelve Columnas
    Lista_num= numeric_df.columns                   #Devuelve lista de Columnas numéricas
    Lista_cat= ['mes','marca','articulo_id']  #Devuelve lista de Columnas categóricas
    #Select box
    Variable_y= st.sidebar.selectbox(label= "Variable dependiente (y)", options= Lista_num)
    #Multi-select box
    Variable_x= st.sidebar.selectbox(label="Variable independiente (x)", options= Lista_cat)  

    #Generamos los encabezados para el dashboard
    st.title("ANOVA")  
   


    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    #Redefinimos las variables 
    Var_num= df[Variable_y]
    Var_cat= df[Variable_x]  
    
    # Contenedor A: ANOVA y resultados
    with Contenedor_A: 
        st.subheader("Resultados del ANOVA")
        grupos = [grupo[Variable_y].dropna() for nombre, grupo in df.groupby(Variable_x)]
        f_valor, p_valor = stats.f_oneway(*grupos)

        st.write(f"**F-valor:** {f_valor:.3f}")
        st.write(f"**p-valor:** {p_valor:.4f}")
        if p_valor < 0.05:
            st.success("¡Hay diferencias significativas entre los grupos!")
        else:
            st.warning("No hay diferencias significativas entre los grupos.")

    # Contenedor B: Gráfico de barras
    with Contenedor_B:
        st.subheader("Gráfico de Barras: Media por grupo")

        resumen = df.groupby(Variable_x)[Variable_y].agg(['mean', 'sem']).reset_index()
        resumen = resumen.sort_values('mean', ascending=False)

        # El deslizador controla cuántos grupos aparecen en la gráfica.
        total_grupos = len(resumen)
        grupos_a_mostrar = st.sidebar.slider(
            label="Número de grupos a mostrar",
            min_value=1,
            max_value=total_grupos,
            value=min(10, total_grupos),
            step=1
        )
        resumen_grafica = resumen.head(grupos_a_mostrar)

        fig, ax = plt.subplots()

        # Dibujo de las barras manualmente con matplotlib para controlar yerr
        x = resumen_grafica[Variable_x].astype(str)  # Asegurar que son strings para el eje x
        y = resumen_grafica['mean']
        yerr = resumen_grafica['sem']

        ax.bar(x, y, yerr=yerr, capsize=5, color='skyblue', edgecolor='black')
        ax.set_ylabel(f"Media de {Variable_y}")
        ax.set_xlabel(Variable_x)
        ax.set_title("Media por grupo con Error Estándar")
        ax.tick_params(axis='x', labelrotation=45)
        fig.tight_layout()
        st.pyplot(fig)
 
