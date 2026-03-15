# -*- coding: utf-8 -*-
# %%
#Ahora si podemos
### Práctica 2: Propiedades estadísticas del lenguaje y Diversidad
#### 1. Verificación empírica de la Ley de Zipf
Verificar si la ley de Zipf se cumple en los siguientes casos:

1.   En un lenguaje artificial creado por ustedes.


*   Creen un script que genere un texto aleatorio seleccionando caracteres al azar de un alfabeto definido


       *   Nota: Asegúrense de incluir el carácter de "espacio" en su alfabeto para que el texto se divida en "palabras" de longitudes variables.


*   Obtengan las frecuencias de las palabras generadas para este texto artificial


2.   Alguna lengua de bajos recursos digitales (low-resourced language)


*  Busca un corpus de libre acceso en alguna lengua de bajos recursos digitales
*   Obten las frecuencias de sus palabras

En ambos casos realiza lo siguiente:


*   Estima el parámetro $\alpha$
 que mejor se ajuste a la curva
*   Generen las gráficas de rango vs. frecuencia (en escala y logarítmica).


    *   Incluye la recta aproximada por $\alpha$
*   ¿Se aproxima a la ley de Zipf? Justifiquen su respuesta comparándolo con el comportamiento del corpus visto en clase.

[!TIP] Puedes utilizar los corpus del paquete py-elotl
"""

# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import string
from collections import Counter
#Aquí colocaré todas las librerías que use para el ejercicio número uno

# %%
#Creamos un minicorpus haciendo uso de random seed para poder replicar el como nos lo fabrica el código para la entrega
random.seed(42)
vocabulario=string.ascii_lowercase + " "#Definimos el vocabulario con ayuda de las letras conocidas a través de string
palabras = random.choices(vocabulario,k=10000)#Formamos las palabras con random choices y formamos 100000 elementos entre esapcios y palabras
corpus="".join(palabras)#Formamos el corpus con dichos 100000 elementos junto a los espacios seleccionados
lista=corpus.split()#Generamos una lista con el corpus generado
df=pd.DataFrame(lista,columns=["palabra"])#Llamamos df a la tabla de las palabras contenidas en el corpus
print(df)#Imprimimos la tabla

# %%
random.seed(42)
vocabulario=string.ascii_lowercase + " "#Definimos el vocabulario con ayuda de las letras conocidas a través de string
def texto(k):
  return "".join(random.choices(vocabulario,k=k))#Formamos el texto con ayuda de random choices
dataset_chiquito=[]
for i in range(10000):
  fila={
      "title": f"Titulo{i}",
      "Texto": texto(10000)
  }
  dataset_chiquito.append(fila)
row=next(iter(dataset_chiquito))
print(row["Texto"])

# %%
df.head(10)#Vemos las primeras palabras dentro de nuestro corpus

# %%
print(row["title"])
print(row["Texto"])

# %%
def counter_to_pandas(counter:Counter)-> pd.DataFrame:
  df=pd.DataFrame.from_dict(counter,orient="index").reset_index()
  df.columnas=["palabra","frecuencia"]
  df.sort_values("count",ascending=False,inplace=True)
  df.reset_index(drop=True,inplace=True)
  return df

# %%
def count_words(corpus)-> Counter:
  count_words=Counter()
  return count_words

# %%
words = count_words(palabras)
words.most_common(10)

# %%
import random
import string

# %%
# 1. Definir el alfabeto (a-z + espacio)
# string.ascii_lowercase es 'abcdefghijklmnopqrstuvwxyz'
alfabeto = string.ascii_lowercase + " "

# %%
# 2. Generar 500 caracteres totalmente al azar
# random.choice elige uno con la misma probabilidad (1/27 cada uno)
texto_lista = [random.choice(alfabeto) for _ in range(500)]

# %%
# 3. Unir y mostrar
texto_final = "".join(texto_lista)
print(texto_final)

# %%
# Si quieres ver las "palabras" que se inventaron:
print("\nPalabras creadas:")
print(texto_final.split())
