################# Variables y tipos de datos :) ###############

#Crear una variable de texto
mi_variable = "Hola amigos"

# Mostrar la varriable de texto
print (mi_variable)

#Crear una lista de números pares
mi_lista = [2, 4, 6, 8, 10]

# Mostrar la lista
print (mi_lista)

#Crear un vector de nombres de planetas para visitar
planetas_para_visitar = ["Marte", "Júpiter", "Saturno", "Urano", "Neptuno"]

# Mostrar el vector de planetas
print (planetas_para_visitar)

#Diccionario con información de la distancia en Km de la Tierra a los planetas para visitar
info_planetas = {
    "Marte": "225000000",
    "Júpiter": "778500000",
    "Saturno": "1427000000",
    "Urano": "2871000000",
    "Neptuno": "4504000000", 
}

# Mostrar el diccionario con informacion de los planetas
print(info_planetas)

############## Tipos de variables: Numéricos, flotantes, complejos #######################

#Creamos vectores con 5 elementos repetidos cada uno
vector_entero_repetido = [42] * 5
vector_flotante_repetido = [2.7878] * 5  # Número con decimales
vector_complejo_repetido = [(3 + 4j)] * 5  # Número complejo

# Crear un diccionario que contenga estos vectores
diccionario = {
    "entero": vector_entero_repetido,
    "flotante": vector_flotante_repetido,
    "complejo": vector_complejo_repetido
}

# Mostrar el diccionario
print(diccionario)

################# Creamos vectores con 5 elementos repetidos cada uno ###################
vector_cadena = ['Hola'] * 5
vector_booleano = [True] * 5
vector_tupla = (7, 'tres') * 5

# Mostrar los vectores resultantes
print("Vector de Cadenas:", vector_cadena)
print("Vector de Booleanos:", vector_booleano)
print("Vector de Tuplas:", vector_tupla)

################### DataFrame ###############

# importar biblioteca
import pandas as pd
# Crear un DataFrame con los datos de la duración estimada de viaje a cada planeta en días 
duracion_viaje_dias = {
    'Planeta': ['Marte', 'Jupiter', 'Saturno', 'Urano', 'Neptuno'],
    'Mision1': [180, 450, 600, 900, 1200],
    'Mision2': [220, 600, 800, 1200, 1500],
    'Mision3': [300, 750, 1000, 1500, 1800],
}

df = pd.DataFrame(duracion_viaje_dias)

# Mostrar el DataFrame
print(df)


########## importar datos de excel ##############
imp_sri = pd.read_excel ("C:/Users/USER/Downloads/clase2 python/ventas_SRI.xlsx",sheet_name="Sheet 1")

# Mostrar datos de excel
print(imp_sri)