#########################################Trabajo Final
#Melissa Chumaña

# Importamos numpy para realizar operaciones numéricas eficientes.
import numpy as np

# Pandas nos permitirá trabajar con conjuntos de datos estructurados.
import pandas as pd

# Desde sklearn.model_selection importaremos funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold

# Utilizaremos sklearn.preprocessing para preprocesar nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler

# sklearn.metrics nos proporcionará métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score

# statsmodels.api nos permitirá realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm

# Por último, matplotlib.pyplot nos ayudará a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt

############################################ Ejercicio 1: Exploración de Datos####################################
##Exploracion de datos
#Variable clave: Tipo_de_piso
#Filtro: sexo=="Mujer"

#Importamos la base de datos
datos = pd.read_csv("sample_endi_model_10p.txt", sep=";")

#Eliminamos las filas con valores nulos en la columna "tipo_de_piso" y "dcronica"
datos = datos[~datos["dcronica"].isna()]
datos = datos[~datos["tipo_de_piso"].isna()]

#Agrupo los datos por tipo de piso
datos.groupby("tipo_de_piso").size()

# Filtrar el DataFrame para obtener solo las filas donde el sexo es 'Mujer'
datos_mujer = datos[datos['sexo'] == 'Mujer']

# Contar cuantas niñas existen por cada categoría de la variable tipo_de_piso'
conteo_por_tipo_de_piso = datos_mujer['tipo_de_piso'].value_counts()
print(conteo_por_tipo_de_piso)

# Calcular el promedio de la variable 'tipo_de_piso' para las niñas
promedio_tipo_de_piso_niñas = datos_mujer['tipo_de_piso'].value_counts().mean()
print("Promedio de la variable 'tipo_de_piso' para las niñas:", promedio_tipo_de_piso_niñas)

########################################################## Ejercicio 2: Modelo Logit#################################################33
#Variables de interes
variables = ['n_hijos', 'region', 'sexo','tipo_de_piso']

#Eliminamos las filas con valores nulos de las variables de interes
for i in variables:
    datos = datos[~datos[i].isna()]

#Cambiamos los codigos numericos en regiones comprensibles
datos["region"] = datos["region"].apply(lambda x: "Costa" if x == 1 else "Sierra" if x == 2 else "Oriente")

#Definimos las variables categoricas y numericas
variables_categoricas = ['region', 'sexo', 'tipo_de_piso']
variables_numericas = ['n_hijos']

#transformador para estandarizar las variables numéricas
transformador = StandardScaler()

#copia de los datos
datos_escalados = datos.copy()

#Estandarizamos las variables numéricas 
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

#convertimos las variables categóricas en variables dummy
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
datos_dummies.columns

#Seleccionamos las variables predictorias y la variable objetivo
#Elimino sexo_mujer porque anteriormente ya se filtro para mujeres y añado las categorias del tipo de piso tomando en cuenta que tipo de piso adecuado es referencia.

X = datos_dummies[['n_hijos', 'region_Sierra','region_Oriente', 
                   'tipo_de_piso_Cemento/Ladrillo', 'tipo_de_piso_Tabla/Caña', 'tipo_de_piso_Tierra']]
y = datos_dummies["dcronica"]

#Definimos los pesos asociados a cada observación
weights = datos_dummies['fexp_nino']

#cantidad de datos que se usaran de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

#Ajustamos el modelo
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)

# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)

# Comparamos las predicciones con los valores reales
predictions_class == y_test

##################################Responder la Pregunta:
#¿Cuál es el valor del parámetro asociado a la variable clave si ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?
#La variable tipo de piso tiene 4 categorias y se toma como categoria de referencia tipo de piso adecuado
#La unica categoria significatuva es tipo_de_piso_Cemento/Ladrillo pues en el modelo es la unica significativa al 1%,10% y 5% con un coeficiente igual a -0.6644

#################################Interpretacion de los resultados
#En base a los resultados se puede decir que respecto a la categoria tipo de piso adecuado, las niñas que tienen tipo de piso cemento/ladrillo 
#tienen menor probabilidad de sufrir desnutrición crónica

####validación cruzada con 100 pliegues

kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

###################Ejercicio 3: Evaluación del Modelo con Datos Filtrados###################

################ Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

#Histograma para visualizar la distribución de las puntuaciones de precisión
plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()


#Histograma para ver la distribución de los coeficientes estimados para la variable “n_hijos”

plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

#Responder a las Preguntas: Responde a las siguientes preguntas en comentarios de tu script: ¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? 
#(Incremento o disminuye ¿Cuanto?)
#La precisión promedio de validación cruzada cuando se utiliza el conjunto de datos filtrado es de 0.6296078431372548 en comparación a la del ejercicio anterior de 0.731372549019608 vemos que disminuye

#¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior? (Incrementa o disminuye ¿Cuanto?)
#Se puede ver que la media de los coeficientes Beta disminuye de un 0.11 a un 0.10   