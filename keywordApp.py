import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from google_play_scraper import app
import re
import string
import unicodedata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


# Título
# Imagem
img = Image. open('kwd_logo.png')
st.image(img)

st.write('''Esta aplicação tem como objetivo encontrar possíveis Keywords e sua densidade
 em textos.''')

# Adicionando link do app e limpando id
# app_link = st.text_input('Insira a url do app que deseja encontrar as keywords e pressione Enter')
# p = re.compile("id=(.*)")
# app_find = p.findall(app_link)

# Buscando informações do app na loja
# result = app(
#     app_find[0],
#     lang='pt',
#     country='br')

# # Criando df para apresentar principais infos sobre o app
# titulo = result['title']
# installs = result['installs']
# score = result['score']
# score = round(float(score),1)
# size = result['size']
# genre = result['genre']
# version = result['version']
# longa = result['description']

# values = [titulo,installs,score,size,genre,version,wds]
# index = ['Título','Instalações','Nota','Tamanho','Categoria','Versão','Palavras na longa']

# info = pd.DataFrame(values, index=index, columns=['Informações do App'])

# st.write(info)

# if st.checkbox('Visualizar Longa Descrição'):
# 	st.write(longa)

longa = st.text_area('Insira um texto:')

########## Infos sobre o texto ##########
wds = len(longa.split())
chrt = len(longa)
chrt_s_space = sum(len(x) for x in longa.split())

values = [wds, chrt, chrt_s_space]
index = ['Palavras','Caracteres','Caracteres s/ espaço']

info = pd.DataFrame(values, index=index, columns=['Informações sobre o Texto'])
st.write(info)


st.subheader('Keywords')
st.write('Selecione o tipo de Keywords que deseja visualizar:')

# Criando lista de stopwords
stops = stopwords.words('portuguese')
stops.extend(['pra','h2','vc','dá','todo','tá','pode','on','line','https','poderá','www','aqui','ainda','confira',
	'sobre','todos','além','assim','e','lá'])

########## Head Tail ##########
co_head = CountVectorizer(ngram_range=(1,1),stop_words=stops)
counts_head = co_head.fit_transform(pd.Series(longa))
headTail = pd.DataFrame(counts_head.sum(axis=0),columns=co_head.get_feature_names()).T.sort_values(0,ascending=False)
headTail.reset_index(inplace=True)
headTail.columns = ['Keyword', 'Count']
# Densidade
headTail['Densidade'] = round(((headTail['Count'] / wds)*100),2)
headTail['Densidade'] = headTail['Densidade'].astype(str)
headTail['Densidade'] = headTail['Densidade'] + '%'

# Visualizar Kwds
if st.checkbox('Head Tail'):
	st.write(headTail)
	st.write('Densidade = Keyword / Número de Palavras')

########## Short Tail ##########
co_short = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(2,2))
counts_short = co_short.fit_transform(pd.Series(longa))
shortTail = pd.DataFrame(counts_short.sum(axis=0),columns=co_short.get_feature_names()).T.sort_values(0,ascending=False)
shortTail.reset_index(inplace=True)
shortTail.columns = ['Keyword', 'Count']

# Retirando stopwords
tmp_short = shortTail["Keyword"].str.split()
shortTail_clean = shortTail[~(tmp_short.str[0].isin(stops) | tmp_short.str[-1].isin(stops))]

# Densidade
shortTail_clean['Densidade'] = round(((shortTail_clean['Count'] / wds)*100),2)
shortTail_clean['Densidade'] = shortTail_clean['Densidade'].astype(str)
shortTail_clean['Densidade'] = shortTail_clean['Densidade'] + '%'

# Visualizar Kwds
if st.checkbox('Short Tail'):
	st.write(shortTail_clean)
	st.write('Densidade = Keyword / Número de Palavras')

########## Long Tail ##########
co = CountVectorizer(token_pattern = r"(?u)\b\w+\b",ngram_range=(3,8))
counts = co.fit_transform(pd.Series(longa))
longTail = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head()
longTail.reset_index(inplace=True)
longTail.columns = ['Keyword', 'Count']

# Retirando stopwords
tmp_long = longTail["Keyword"].str.split()
longTail_clean = longTail[~(tmp_long.str[0].isin(stops) | tmp_long.str[-1].isin(stops))]

# Densidade
longTail_clean['Densidade'] = round(((longTail_clean['Count'] / wds)*100),2)
longTail_clean['Densidade'] = longTail_clean['Densidade'].astype(str)
longTail_clean['Densidade'] = longTail_clean['Densidade'] + '%'

# Visualizar Kwds
if st.checkbox('Long Tail'):
	st.write(longTail_clean)
	st.write('Densidade = Keyword / Número de Palavras')