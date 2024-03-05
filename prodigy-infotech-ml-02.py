#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


df=pd.read_csv("Mall_Customers.csv")


# In[3]:


df.head()


# In[4]:


df.drop("CustomerID",inplace=True,axis=1)


# In[5]:


df.isna().sum()


# In[6]:


df.shape


# In[7]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="rocket", annot=True, fmt='.3f',annot_kws={"size": 12})
plt.show()


# In[8]:


df.describe()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le = LabelEncoder()


# In[11]:


df["Gender"] = le.fit_transform(df.Gender)


# In[12]:


df.head()


# In[13]:


plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.show()


# In[49]:


plt.figure(figsize=(15, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df, s=100, alpha=0.7)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.show()


# In[14]:


X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)


# In[15]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


ilist = []  # Within-Cluster-Sum-of-Squares
    
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    ilist.append(kmeans.inertia_)

# Plot the Elbow method
plt.plot(range(1, 11), ilist,marker='o', linestyle='--', color='g')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('ILIST')  # Within-Cluster-Sum-of-Squares
plt.show()


# In[17]:


# Training the K-Means Clustering Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)


# In[18]:


# Apply K-Means clustering
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_train_scaled)


# In[19]:


df.head()


# In[20]:


kmeans.inertia_


# In[37]:


y_means=kmeans.fit_predict(X)


# In[46]:


plt.scatter(X.iloc[y_means==0,0], X.iloc[y_means==0,0], s = 50,c = 'red', label="Cluster 1" )
plt.scatter(X.iloc[y_means==1,0], X.iloc[y_means==1,1], s = 50,c = 'blue', label="Cluster 2" )
plt.scatter(X.iloc[y_means==2,0], X.iloc[y_means==2,1], s = 50,c = 'black', label="Cluster 3" )
plt.scatter(X.iloc[y_means==3,0], X.iloc[y_means==3,1], s = 50,c = 'green', label="Cluster 4" )
plt.scatter(X.iloc[y_means==4,0], X.iloc[y_means==4,1], s = 50,c = 'orange', label="Cluster 5" )
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=10,c= "magenta")
plt.title("Customer Segementation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()


# In[51]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[[ 'Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[52]:


inertia_scores2=[]
for i in range(1,11) :
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)


# In[53]:


centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']
centers


# In[59]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='r',marker='*')
sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)', hue='Spending and Income Cluster',palette ="tab10");


# In[ ]:




