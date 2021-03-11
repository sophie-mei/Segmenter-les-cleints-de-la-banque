#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas
import tensorflow
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Add, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import  Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import SGD
#import _pywrap_tensorflow_internal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pandas.read_csv('segmentation.csv')
data.head()
data.info()
data.nunique()
data.describe().T
nt de 50 dollars. En moyenne, les clients ont une limite de carte de crédit de 4494 dollars.
data[data['CASH_ADVANCE']==47137.211760000006]
data.isnull().sum()

def valeur_manquante_pourcentage(df):

    total = data.isnull().sum().sort_values(
        ascending=False)[data.isnull().sum().sort_values(ascending=False) != 0]
    percent = (data.isnull().sum().sort_values(ascending=False) / len(data) *
               100)[(data.isnull().sum().sort_values(ascending=False) / len(data) *
                     100) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

valeur_manquante_data = valeur_manquante_pourcentage(data)

fig, ax = plt.subplots( figsize=(16,8))

sns.barplot(x=valeur_manquante_data.index,
            y='Percent',
            data=valeur_manquante_data)


ax.set_title('Valeur Manquante')
plt.show()
valeur_manquante_data
valeur_manquante_data.to_csv("valeur_manquante_data.csv",index=False)
valeur_manquante_data.head()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(data.MINIMUM_PAYMENTS, color="#fdc029")
plt.subplot(1,2,2)
sns.distplot(data.CREDIT_LIMIT, color="#fdc029")
plt.show()

#après avoir pris la conaissance de la distribution des deux variables, on va remplacer les valeurs manquantes par la moyenne
data.MINIMUM_PAYMENTS.fillna(data.MINIMUM_PAYMENTS.mean(),inplace=True)
data.CREDIT_LIMIT.fillna(data.CREDIT_LIMIT.mean(),inplace=True)

#verifions si il nous reste encore des valeurs manquantes dans les données 
print('la variable MINIMUM_PAYMENTS', data.MINIMUM_PAYMENTS.isnull().sum(),'valeur manquante')
print('la variable CREDIT_LIMIT', data.CREDIT_LIMIT.isnull().sum(),'valeur manquante')
print('la variable CREDIT_LIMIT', data.CREDIT_LIMIT.isnull().sum(),'valeur manquante')

#verifions les doublons 
data.duplicated().sum()

data.drop("CUST_ID",axis=1,inplace=True)
# L'denfication personnalisée n'est pas une information inutile qui ne ferait qu'ajouter de la complexité aux données, car il 
#s'agit d'un objet et non une information numérique. Donc cette information sera supprimé de l'ensemblle de donnée. 

data.to_csv("segmenation_propre.csv", index=False)
data.head()

plt.figure(figsize=(10,10))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.show()

#Dans cette étude de cas, il est préferable de ne pas traiter les valeurs aberrantes, car nous voulons analyser tous les types 
#de clients, donc il vaut mieux ne pas les traiter

import numpy as np
def correlation(df):
    
    corr = abs(data.corr()) # correlation matrix
    infe_triangle = np.tril(corr, k = -1)  # ne sélectionner que le triangle inférieur de la matrice de corrélation
    mask = infe_triangle == 0 # pour masquer le triangle supérieur dans la carte thermique suivante

    plt.figure(figsize = (15,10))  
    sns.set_style(style = 'white')  # Le mettre en blanc pour qu'on ne voit pas les lignes de la grille
    sns.heatmap(infe_triangle, center=0.5, cmap= 'Blues', xticklabels = corr.index,
                yticklabels = corr.columns,cbar = False, annot= True, linewidths= 1, mask = mask)   
    plt.show()
    
correlation(data)

sns.set_palette('Set2')
sns.set_style("dark")
data.hist(bins=40, figsize=(20,20))

#commencons par mettre  les données à l'echelle 
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_data=scaler.fit_transform(dataset)
scaled_data=pd.DataFrame(data=scaled_data,columns = [dataset.columns])

from sklearn.cluster import KMeans

scores_1= []

range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    scores_1.append(kmeans.inertia_)
plt.plot(scores_1, 'bx-')
plt.style.use('ggplot')
plt.title('trouver le bon cluster')
plt.xlabel('Clusters')
plt.ylabel('Variance intra-cluster totale') 
plt.show()

kmeans = KMeans(8)
kmeans.fit(scaled_data)
labels = kmeans.labels_

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_,columns = [data.columns])

#Afin de comprendre la significatif de ces chiffre, effectuons une transformation inverse
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers,columns = [data.columns])
cluster_centers.to_csv("cluster_centers_1.csv",index=False)

y_kmeans = kmeans.fit_predict(scaled_data)

# Concatener les cluster avec notre data 
creditcard_df_cluster = pd.concat([data, pd.DataFrame({'cluster':labels})], axis = 1)

for i in data.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()

#ACP permet de réduire les dimensions tout en réserve les mêmes informations 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(scaled_data)

#Créer un dataframe avec 2 composantes 
pca_df = pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])
pca_df.sample(5)

pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()

plt.figure(figsize=(10,10))
plt.style.use('ggplot')
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
plt.show()

encoding_dim = 7

input_df = Input(shape=(17,))


#
x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)

#encoder - utilisé pour notre réduction de la dimention
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')

autoencoder.fit(scaled_data,scaled_data,batch_size=128,epochs=25,verbose=1)

autoencoder.summary()

pred_ac = encoder.predict(scaled_data)

pred_ac = pd.DataFrame(data=pred_ac)

scores_2 = []

range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pred_ac)
    scores_2.append(kmeans.inertia_)
plt.plot(scores_2, 'bx-')
plt.style.use('ggplot')
plt.title('Trouver le nombre clusters')
plt.xlabel('Clusters')
plt.ylabel('Variance intra-cluster totale') 
plt.show()

plt.plot(scores_1, 'bx-', color = 'r',label='Without Autoencode')
plt.plot(scores_2, 'bx-', color = 'g',label='With Autoencode')

kmeans = KMeans(4)
kmeans.fit(pred_ac)
labels = kmeans.labels_
kmeans.cluster_centers_.shape

y_kmeans = kmeans.fit_predict(scaled_data)

# concaténer les nouveaux labels de clusters réduits à notre cadre de données initial
creditcard_df_cluster_new = pd.concat([data, pd.DataFrame({'cluster':labels})], axis = 1)

# Plot the histogram of various clusters
for i in data.columns:
  plt.figure(figsize = (20, 5))
  for j in range(4):
    plt.subplot(1,4,j+1)
    cluster = creditcard_df_cluster_new[creditcard_df_cluster_new['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()

pca = PCA(n_components=2)
principal_comp_new = pca.fit_transform(pred_ac)

# Create a dataframe with the two components
pca_df = pd.DataFrame(data=principal_comp_new,columns=['pca1','pca2'])

pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)

plt.figure(figsize=(30,30))
plt.style.use('ggplot')
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink'])
plt.show()

