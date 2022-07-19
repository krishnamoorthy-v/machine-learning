from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("/home/krishna/Desktop/csv.csv")

plt.scatter(df["Age"], df["Income($)"])
plt.xlabel("age")
plt.ylabel("income")

km = KMeans(n_clusters= 3)
y_predicted = km.fit_predict(df[['Age', "Income($)"]])
df['cluster'] = y_predicted

df1 = df[df["cluster"]==0]
df2 = df[df["cluster"]==1]
df3 = df[df["cluster"]==2]
plt.scatter(df1["Age"], df1["Income($)"], color="green")
plt.scatter(df2["Age"], df2["Income($)"], color="red")
plt.scatter(df3["Age"], df3["Income($)"], color="blue")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="purple", marker="*", label="centroid")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()

scalar = MinMaxScaler()
scalar.fit(df[["Age"]])
df["Age"] = scalar.transform(df[["Age"]])

scalar.fit(df[['Income($)']])
df["Income($)"] = scalar.transform(df[["Income($)"]])

plt.scatter(df["Age"], df["Income($)"])
km = KMeans(n_clusters= 3)
y_predicted = km.fit_predict(df[['Age', "Income($)"]])
df['cluster'] = y_predicted

df1 = df[df["cluster"]==0]
df2 = df[df["cluster"]==1]
df3 = df[df["cluster"]==2]
plt.scatter(df1["Age"], df1["Income($)"], color="green", label="cluster1")
plt.scatter(df2["Age"], df2["Income($)"], color="red", label="cluster2")
plt.scatter(df3["Age"], df3["Income($)"], color="blue", label="cluster3")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color="purple", marker="*", label="centroid")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()