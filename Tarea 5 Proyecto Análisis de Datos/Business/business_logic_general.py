# Importo mi Clase con las constantes declaradas con las rutas y algunos mensajes que utilizare
from Business.constants import MagicString

# Librerias para analisis Exploratorio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Importo las funcionalidades necesarias para realizar la implementacion de los modelos de ciencia de datos
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, RocCurveDisplay

import warnings
warnings.filterwarnings('ignore')

class General:
    def leer_datos_csv(ruta : str):
        Datos = pd.read_csv(ruta)

        # Remuevo registros que puedan estar duplicados
        Datos.drop_duplicates(inplace=True)

        return Datos
    

    def realizar_lectura_datos_analisis(datos : pd.DataFrame, cantidad_registros : int):
        return datos.head(cantidad_registros)


    def describir_datos_csv(datos: pd.DataFrame):
        return datos.describe()
    
    
    def graficar_datos_atipicos(Datos: pd.DataFrame, columna : str):
        plt.figure(figsize=(6,3))
        sns.boxplot(x=Datos[columna])
        plt.title(MagicString.TITLE_VALORES_ATIPICOS.format(columna),fontsize=10)
    

    def genera_grafica_de_correlacion(datos : pd.DataFrame):
        plt.figure(figsize=(10,10))
        sns.heatmap(datos.corr(), annot=True, cmap='coolwarm')
        plt.title(MagicString.MATRIZ_CORRELACION_TITLE)
        plt.show()


    def modificar_columna_genero(Datos: pd.DataFrame, columna : str):
        Datos[columna] = Datos[columna].replace('male', 0)
        Datos[columna] = Datos[columna].replace('female', 1)

        return Datos[columna]
    
    def modificar_columna_embarco(Datos: pd.DataFrame, columna):
        Datos[columna] = Datos[columna].replace('C', 0)
        Datos[columna] = Datos[columna].replace('S', 1)
        Datos[columna] = Datos[columna].replace('Q', 2)

        return Datos[columna]
    
    def modificar_columna_fare(Datos: pd.DataFrame, columna):
        Datos[columna] = Datos[columna].replace(0, Datos[columna].mean())
        
        return Datos[columna]


    def realizar_imputacion_de_datos_para_datos_atipicos(datos : pd.DataFrame, nombre_columna: str) -> pd.DataFrame:
        # Calculo el Quartile 1 y Quartile 3
        Q1 = datos[nombre_columna].quantile(0.25)
        Q3 = datos[nombre_columna].quantile(0.75)
        IQR = Q3 - Q1

        #defino los limites de mis datos.
        limite_inferior = Q1 - 1 * IQR
        limite_superior = Q3 + 1 * IQR

        # Calcular la mediana
        mediana = datos[nombre_columna].median()

        # Imputar outliers con la mediana
        return  datos[nombre_columna].mask((datos[nombre_columna] < limite_inferior) | (datos[nombre_columna] > limite_superior), mediana)


    def dividir_registros_train_y_test(datos : pd.DataFrame, header : str, state):
        X = datos.drop(header, axis=1)
        Y = datos[header]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=state)

        return x_train, x_test, y_train, y_test
    

    def entrenar_modelo_regresion_logistica(x_train, y_train, x_test):
        regresion_logistica_modelo = LogisticRegression(solver='liblinear')
        regresion_logistica_modelo.fit(x_train, y_train)

        return regresion_logistica_modelo.predict(x_test)    
    
    
    def evaluar_desempeño_modelo_presicion(modelo, y_test):
        print(classification_report(y_test, modelo,zero_division=0))
    
    
    def matriz_de_confusion(y_test, prediccion):
        # Generar grafica matriz de confusion
        matriz_confusion = confusion_matrix(y_test, prediccion)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix = matriz_confusion)

        return disp.plot()
    
    
    def curva_de_precision_modelo(y_test, prediccion, title):
        RocCurveDisplay.from_predictions(y_test, prediccion)
        plt.title(title, fontsize=25)
        plt.show()

    
    def generar_arbol_de_decision(x_train, y_train, x_test, y_test):
        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        modelo = tree.predict(x_test)
        
        General.evaluar_desempeño_modelo_presicion(modelo, y_test)

        plot_tree(tree)