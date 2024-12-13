import pandas as pd
import re

import regex
import demoji

import numpy as np
from collections import Counter

import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

import streamlit as st

###################################
###################################
# Título de la aplicación
st.title('Análisis de nuestro chat de WhatsApp Valeria Montaño 💗')
st.write('Creado por [Yoseth](https://drive.google.com/file/d/1aKo1TG4AbKO7YBRsj6xRke5YKxkkRfYu/view?usp=sharing) para la novia más perfecta, hermosa y comprensiva del mundo, [Valeria](https://drive.google.com/file/d/1JXkucKKR48Gnok7IVz5aESi_AreSE43q/view?usp=sharing)')
st.write('Valeria, eres una persona increíblemente fuerte y perseverante. Cuidadosa con tu familia, llena de empatía por los demás y con un corazón que ama de una manera única. Eres mi mayor fuente de inspiración y no hay palabras suficientes para expresar cuánto te adoro. Te amo profundamente y quiero estar a tu lado, tomando tu mano en cada logro y cada paso que des en esta vida.')
###################################
###################################

##########################################
# ### Paso 1: Definir funciones necesarias
##########################################

# Patron regex para identificar el comienzo de cada línea del txt con la fecha y la horadef arranque(s):
def arranque(s):
    patron = r"(\d{1,2}/\d{2}/\d{2}, \d{1,2}:\d{2} [AP][M] - .+?)(?=\s+\d{1,2}/\d{2}/\d{2}, \d{1,2}:\d{2} [AP][M] - |\Z)"
    resultado = re.findall(patron, s)  # Verificar si cada línea del txt hace match con el patrón de fecha y hora
    return resultado

# Patrón para encontrar a los miembros del grupo dentro del txt

def EncontrarMiembro(s):
    patron= r"Yoseth 🐣:|Valeria 💗:"
    resultado = re.match(patron, s)  # Verificar si cada línea del txt hace match con el patrón de miembro
    if resultado:
        return True
    return False

# Separar las partes de cada línea del txt: Fecha, Hora, Miembro y Mensaje
def ObtenerPartes(linea):
    # Ejemplo: '10/05/23, 9:06 AM - Valeria 💗: Holaaa Yoseth, cómo vas?'
    splitLinea = linea.split(' - ')
    FechaHora = splitLinea[0]                     # '10/05/23, 9:06 AM'
    splitFechaHora = FechaHora.split(', ')
    Fecha = splitFechaHora[0]                    # '10/05/23'
    Hora = ' '.join(splitFechaHora[1:])          # '9:06 AM'
    Mensaje = ' '.join(splitLinea[1:])             # 'Sandreke: Todos debemos aprender a analizar datos'
    if EncontrarMiembro(Mensaje):
        splitMensaje = Mensaje.split(': ')
        Miembro = splitMensaje[0]               # 'Valeria 💗:'
        Mensaje = ' '.join(splitMensaje[1:])    # 'Holaaa Yoseth, cómo vas?'
    else:
        Miembro = None
    return Fecha, Hora, Miembro, Mensaje


##################################################################################
# ### Paso 2: Obtener el dataframe usando el archivo txt y las funciones definidas
##################################################################################

# Leer el archivo txt descargado del chat de WhatsApp
RutaChat = 'Chat de WhatsApp con Valeria 💗.txt'

# Leer el archivo
with open(RutaChat, "r") as archivo:
    contenido = archivo.read()

# 1era limpieza del contendio
contenido=contenido.replace("\u202f"," ").replace("a. m.","AM").replace("p. m.", "PM").replace("Arciniegas B. Yoseth:","Yoseth 🐣:")
contenido_limpio=contenido.replace("\n"," ")
# SEPARAMOS CADA TEXT POR FECHAS, SACAMOS LAS DOS PRIMERAS LINEAS COMO TAMBIEN SACAMOS MENSAJES DE WPP
mensajes_limpios=arranque(contenido_limpio)
del mensajes_limpios[8778] # este es un mensaje de 'fijaste un mensaje'
del mensajes_limpios[0:2]
# OBTENEMOS LAS PARTES DE CADA UNO DE LOS MENSAJES LIMPIOS
DatosLista=[]
for elemento in mensajes_limpios:
  DatosLista.append(ObtenerPartes(elemento))

df = pd.DataFrame(DatosLista, columns=['Fecha', 'Hora', 'Miembro', 'Mensaje'])
# Cambiar la columna Fecha a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%y',  errors='coerce').dt.strftime('%Y-%m-%d')

# Eliminar los posibles campos vacíos del dataframe
# y lo que no son mensajes como cambiar el asunto del grupo o agregar a alguien
df = df.dropna()

# Resetear el índice
df.reset_index(drop=True, inplace=True)

##################################################################
# ### Paso 3: Estadísticas de mensajes, multimedia, emojis y links
##################################################################

# #### Total de mensajes, multimedia, emojis y links
def ObtenerEmojis(Mensaje):
    emoji_lista = []
    data = regex.findall(r'\X', Mensaje)  # Obtener lista de caracteres de cada mensaje
    for caracter in data:
        if demoji.replace(caracter) != caracter:
            emoji_lista.append(caracter)
    return emoji_lista

# Obtener la cantidad total de mensajes
total_mensajes = df.shape[0]

# Obtener la cantidad de archivos multimedia enviados
multimedia_mensajes = df[df['Mensaje'] == '<Multimedia omitido>'].shape[0]

# Obtener la cantidad de emojis enviados
df['Emojis'] = df['Mensaje'].apply(ObtenerEmojis) # Se agrega columna 'Emojis'
emojis = sum(df['Emojis'].str.len())

# Obtener la cantidad de links enviados
url_patron = r'(https?://\S+)'
df['URLs'] = df.Mensaje.apply(lambda x: len(re.findall(url_patron, x))) # Se agrega columna 'URLs'
links = sum(df['URLs'])

# Obtener la cantidad de encuestas
encuestas = df[df['Mensaje'] == 'POLL:'].shape[0]

# Todos los datos pasarlo a diccionario
estadistica_dict = {'Tipo': ['Mensajes', 'Multimedia', 'Emojis', 'Links', 'Encuestas'],
        'Cantidad': [total_mensajes, multimedia_mensajes, emojis, links, encuestas]
        }

#Convertir diccionario a dataframe
estadistica_df = pd.DataFrame(estadistica_dict, columns = ['Tipo', 'Cantidad'])

# Establecer la columna Tipo como índice
estadistica_df = estadistica_df.set_index('Tipo')

###################################
###################################
st.header('💡 Estadísticas generales')
col1, col2 = st.columns([1, 2])

with col1:
    st.write(estadistica_df)
###################################
###################################

# #### Emojis más usados

# Obtener emojis más usados y las cantidades en el chat del grupo del dataframe
emojis_lista = list([a for b in df.Emojis for a in b])
emoji_diccionario = dict(Counter(emojis_lista))
emoji_diccionario = sorted(emoji_diccionario.items(), key=lambda x: x[1], reverse=True)

# Convertir el diccionario a dataframe
emoji_df = pd.DataFrame(emoji_diccionario, columns=['Emoji', 'Cantidad'])

# Establecer la columna Emoji como índice
emoji_df = emoji_df.set_index('Emoji').head(10)

# Plotear el pie de los emojis más usados
fig = px.pie(emoji_df, values='Cantidad', names=emoji_df.index, hole=.3, template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20)

# Ajustar el gráfico
fig.update_layout(
    # title={'text': 'Emojis más usados',
    # #          'y':0.96,
    # #          'x':0.5,
    #          'xanchor': 'center'}, font=dict(size=17),
    showlegend=False)
# fig.show()



###################################
###################################
# st.header('Emojis más usados')
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.write(emoji_df)

# with col2:
#     st.plotly_chart(fig)
###################################
###################################

# ### Paso 4: Estadísticas de los miembros del grupo

# #### Miembros más activos

# Determinar los miembros más activos del grupo
df_MiembrosActivos = df.groupby('Miembro')['Mensaje'].count().sort_values(ascending=False).to_frame()
df_MiembrosActivos.reset_index(inplace=True)
df_MiembrosActivos.index = np.arange(1, len(df_MiembrosActivos)+1)
df_MiembrosActivos['% Mensaje'] = (df_MiembrosActivos['Mensaje'] / df_MiembrosActivos['Mensaje'].sum()) * 100



###################################
###################################
with col2:
    st.write(df_MiembrosActivos)
###################################
###################################

# #### Estadísticas por miembro

# Separar mensajes (sin multimedia) y multimedia (stickers, fotos, videos)
multimedia_df = df[df['Mensaje'] == '<Multimedia omitido>']
mensajes_df = df.drop(multimedia_df.index)

# Contar la cantidad de palabras y letras por mensaje
mensajes_df['Letras'] = mensajes_df['Mensaje'].apply(lambda s : len(s))
mensajes_df['Palabras'] = mensajes_df['Mensaje'].apply(lambda s : len(s.split(' ')))
# mensajes_df.tail()


# Obtener a todos los miembros
miembros = mensajes_df.Miembro.unique()

# Crear diccionario donde se almacenará todos los datos
dictionario = {}

for i in range(len(miembros)):
    lista = []
    # Filtrar mensajes de un miembro en específico
    miembro_df= mensajes_df[mensajes_df['Miembro'] == miembros[i]]

    # Agregar a la lista el número total de mensajes enviados
    lista.append(miembro_df.shape[0])
    
    # Agregar a la lista el número de palabras por total de mensajes (palabras por mensaje)
    palabras_por_msj = (np.sum(miembro_df['Palabras']))/miembro_df.shape[0]
    lista.append(palabras_por_msj)

    # Agregar a la lista el número de mensajes multimedia enviados
    multimedia = multimedia_df[multimedia_df['Miembro'] == miembros[i]].shape[0]
    lista.append(multimedia)

    # Agregar a la lista el número total de emojis enviados
    emojis = sum(miembro_df['Emojis'].str.len())
    lista.append(emojis)

    # Agregar a la lista el número total de links enviados
    links = sum(miembro_df['URLs'])
    lista.append(links)

    # Asignar la lista como valor a la llave del diccionario
    dictionario[miembros[i]] = lista
    
# print(dictionario)


# Convertir de diccionario a dataframe
miembro_stats_df = pd.DataFrame.from_dict(dictionario)

# Cambiar el índice por la columna agregada 'Estadísticas'
estadísticas = ['Mensajes', 'Palabras por mensaje', 'Multimedia', 'Emojis', 'Links']
miembro_stats_df['Estadísticas'] = estadísticas
miembro_stats_df.set_index('Estadísticas', inplace=True)

# Transponer el dataframe
miembro_stats_df = miembro_stats_df.T

#Convertir a integer las columnas Mensajes, Multimedia Emojis y Links
miembro_stats_df['Mensajes'] = miembro_stats_df['Mensajes'].apply(int)
miembro_stats_df['Multimedia'] = miembro_stats_df['Multimedia'].apply(int)
miembro_stats_df['Emojis'] = miembro_stats_df['Emojis'].apply(int)
miembro_stats_df['Links'] = miembro_stats_df['Links'].apply(int)
miembro_stats_df = miembro_stats_df.sort_values(by=['Mensajes'], ascending=False)

###################################
###################################
st.subheader('Cómo se distribuyen nuestros mensajes 👀')
st.write(miembro_stats_df)
###################################
###################################



###################################
###################################
st.header('🤗 Emojis más usados')
col1, col2 = st.columns([1, 2])

with col1:
    st.write(emoji_df)

with col2:
    st.plotly_chart(fig)
###################################
###################################



# ### Paso 5: Estadísticas del comportamiento del grupo

df['rangoHora'] = pd.to_datetime(df['Hora'], format='%I:%M %p')

# Define a function to create the "Range Hour" column
def create_range_hour(hour):
    hour = pd.to_datetime(hour)  # Convertir a objeto de Python datetime si es necesario
    start_hour = hour.hour
    end_hour = (hour + pd.Timedelta(hours=1)).hour
    return f'{start_hour:02d} - {end_hour:02d} h'

# # Apply the function to create the "Range Hour" column
df['rangoHora'] = df['rangoHora'].apply(create_range_hour)
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
# Obtener el nombre del día de la semana
df['DiaSemana'] = df['Fecha'].dt.strftime('%A')
# Mapeo de los días a español
mapeo_dias_espanol = {
    'Monday': '1 Lunes', 'Tuesday': '2 Martes', 'Wednesday': '3 Miércoles',
    'Thursday': '4 Jueves', 'Friday': '5 Viernes', 'Saturday': '6 Sábado',
    'Sunday': '7 Domingo'
}
# Aplicar el mapeo
df['DiaSemana'] = df['DiaSemana'].map(mapeo_dias_espanol)




# #### Número de mensajes por rango de hora

# Crear una columna de 1 para realizar el conteo de mensajes
df['# Mensajes por hora'] = 1

# Sumar (contar) los mensajes que tengan la misma fecha
date_df = df.groupby('rangoHora').count().reset_index()

# Plotear la cantidad de mensajes respecto del tiempo
fig = px.line(date_df, x='rangoHora', y='# Mensajes por hora', color_discrete_sequence=['salmon'], template='plotly_dark')

# Ajustar el gráfico
# fig.update_layout(
#     title={'text': 'Mensajes con ella ❤️ por hora',
#              'y':0.96,
#              'x':0.5,
#              'xanchor': 'center'},
#     font=dict(size=17))
fig.update_traces(mode='markers+lines', marker=dict(size=10))
fig.update_xaxes(title_text='Rango de hora', tickangle=30)
fig.update_yaxes(title_text='# Mensajes')
# fig.show()

###################################
###################################
st.header('⏰ Mensajes por hora')
st.plotly_chart(fig)
###################################
###################################


# #### Número de mensajes por día

# Crear una columna de 1 para realizar el conteo de mensajes
df['# Mensajes por día'] = 1

# Sumar (contar) los mensajes que tengan la misma fecha
date_df = df.groupby('DiaSemana').count().reset_index()


# Plotear la cantidad de mensajes respecto del tiempo
fig = px.line(date_df, x='DiaSemana', y='# Mensajes por día', color_discrete_sequence=['salmon'], template='plotly_dark')

# Ajustar el gráfico
# fig.update_layout(
#     title={'text': 'Mensajes con ella ❤️ por día', 'y':0.96, 'x':0.5, 'xanchor': 'center'},
#     font=dict(size=17))

fig.update_traces(mode='markers+lines', marker=dict(size=10))
fig.update_xaxes(title_text='Día', tickangle=30)
fig.update_yaxes(title_text='# Mensajes')
# fig.show()

###################################
###################################
st.header('📆 Mensajes por día')
st.plotly_chart(fig)
###################################
###################################

# #### Número de mensajes a través del tiempo

# Crear una columna de 1 para realizar el conteo de mensajes
df['# Mensajes por día'] = 1

# Sumar (contar) los mensajes que tengan la misma fecha
date_df = df.groupby('Fecha').sum().reset_index()

# Plotear la cantidad de mensajes respecto del tiempo
fig = px.line(date_df, x='Fecha', y='# Mensajes por día', color_discrete_sequence=['salmon'], template='plotly_dark')

# Ajustar el gráfico
# fig.update_layout(
#     title={'text': 'Mensajes con ella ❤️',
#              'y':0.96,
#              'x':0.5,
#              'xanchor': 'center'},
#     font=dict(size=17))

fig.update_xaxes(title_text='Fecha', tickangle=45, nticks=35)
fig.update_yaxes(title_text='# Mensajes')
# fig.show()

###################################
###################################
st.header('📈 Mensajes a lo largo del tiempo')
st.plotly_chart(fig)
###################################
###################################

# #### Word Cloud de palabras más usadas

# Crear un string que contendrá todas las palabras
total_palabras = ' '
stopwords = STOPWORDS.update(['que', 'qué', 'con', 'de', 'te', 'en', 'la', 'lo', 'le', 'el', 'las', 'los', 'les', 'por', 'es',
                              'son', 'se', 'para', 'un', 'una', 'chicos', 'su', 'si', 'chic','nos', 'ya', 'hay', 'esta',
                              'pero', 'del', 'mas', 'más', 'eso', 'este', 'como', 'así', 'todo', 'https','Multimedia','omitida',
                              'y', 'mi', 'o', 'q', 'yo', 'al', 'voy', 'null', 'pues','porque','editó'])

mask = np.array(Image.open('heart.jpg'))

# Obtener y acumular todas las palabras de cada mensaje
for mensaje in mensajes_df['Mensaje'].values:
    palabras = str(mensaje).lower().split() # Obtener las palabras de cada línea del txt
    for palabra in palabras:
        total_palabras = total_palabras + palabra + ' ' # Acumular todas las palabras

wordcloud = WordCloud(width = 800, height = 800, background_color ='black', stopwords = stopwords,
                      max_words=100, min_font_size = 5,
                      mask = mask, colormap='OrRd',).generate(total_palabras)

# Plotear la nube de palabras más usadas
# wordcloud.to_image()


###################################
###################################
st.header('☁️ Nuestro word cloud')
st.image(wordcloud.to_array(), caption='Las palabras que más usamos ❤️', use_column_width=True)
###################################
###################################
