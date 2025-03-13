
from flask import Flask
from flaskext.mysql import MySQL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.pyplot import ylabel
from scipy.stats import alpha
from sklearn.cluster import KMeans
import numpy as np
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

def getConnect(server, port, database, username, password):
    try:
        mysql = MySQL()

        # MySQL configurations
        app.config['MYSQL_DATABASE_HOST'] = server
        app.config['MYSQL_DATABASE_PORT'] = port
        app.config['MYSQL_DATABASE_DB'] = database
        app.config['MYSQL_DATABASE_USER'] = username
        app.config['MYSQL_DATABASE_PASSWORD'] = password

        mysql.init_app(app)
        conn = mysql.connect()
        return conn
    except Error as e:
        print("Error = ", e)
    return None

def closeConnection(conn):
    if conn is not None:
        conn.close()

def queryDataset(conn,sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall())
    return df

conn = getConnect('localhost',3306,'sakila','root','Huygia@123')
sql1 ="select * from customer"
df1 = queryDataset(conn, sql1)
print(df1)

sql2 = """
SELECT DISTINCT 
    c.customer_id, 
    f.rental_rate,
    f.length,
    f.rental_duration
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN film f ON i.film_id = f.film_id;
"""
df2 = queryDataset(conn, sql2)
df2.columns = ['customer_id', 'rental_rate','length','rental_duration']

print(df2)
print(df2.head())
print(df2.describe())

def showHistogram(df,columns):
    plt.figure(1,figsize=(7,8))
    n =0
    for column in columns:
        n +=1
        plt.subplot(3,1,n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.displot(df[column],bins =32)
        plt.title(f'Histogram of {column}')
    plt.show()

showHistogram(df2,df2.columns[1:])

def elbowMethod(df,columnsForElbow):
    X = df.loc[:,columnsForElbow].values
    inertia = []
    for n in range(1,11):
        model = KMeans(n_clusters=n,
                       init= 'k-means++',
                       max_iter=500,
                       random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
    plt.figure(1,figsize=(15,16))
    plt.plot(np.arange(1,11),inertia,'o')
    plt.plot(np.arange(1,11), inertia, '-',alpha = 0.5)
    plt.xlabel('Number of Clusters'), plt.ylabel('Clusers sum of squared distance')
    plt.show()

columns = ['length','rental_rate']
elbowMethod(df2,columns)

def runKMeans(X,cluster):
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   random_state=42)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)
    return y_kmeans,centroids,labels

X = df2.loc[:, columns].values
cluster = 3
colors = ['green','red','blue','purple','black','pink','orange']

y_kmeans, centroids, labels = runKMeans(X,cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2['cluster'] = labels


def visualizeKMeans(X,y_kmeans,cluster,title,xlabel,ylabel,colors):
    plt.figure(figsize=(10,10))
    for i in range (cluster):
        plt.scatter(X[y_kmeans==i,0],
                    X[y_kmeans==i,1],
                    s=100,
                    c = colors[i],
                    label = 'Cluster%i'%(i+1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

visualizeKMeans(X,y_kmeans,cluster,'Clustoer of Custoemr - length x rental_rate','length','rental_rate',colors)

columns = ['length','rental_duration']
elbowMethod(df2,columns)

X = df2.loc[:,columns].values
cluster = 2

y_kmeans,centroids,labels = runKMeans(X,cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2['cluster'] = labels

visualizeKMeans(X,y_kmeans,cluster,'Clustoer of Customer - length x rental_duration','length','rental_duration',colors)

columns = ['rental_rate','rental_duration','length']
elbowMethod(df2,columns)
X = df2.loc[:,columns].values
cluster = 2

y_kmeans,centroids,labels = runKMeans(X,cluster)
print(y_kmeans)
print(centroids)
print(labels)
df2['cluster'] = labels
print(df2)

def visualize3DKMeans(df,columns,hover_data,cluster):
    fig = px.scatter_3d(df,
                        x=columns[0],
                        y=columns[1],
                        z=columns[2],
                        color = 'cluster',
                        hover_data=hover_data,
                        category_orders={"cluster":range(0,cluster)},)
    fig.update_layout(margin = dict(l=0,r=0,t=0))
    fig.show()

hover_data=df2.columns
visualize3DKMeans(df2,columns, hover_data, cluster)