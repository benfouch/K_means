from sklearn.cluster import KMeans
import pandas as pd

master_data         = pd.read_csv('iris.csv')
master_data         = master_data.sample(frac=1).reset_index(drop=True)

key_data            = master_data['Species']
train_test          = master_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',  'PetalWidthCm']]

train_df            = train_test[:int(.80 * train_test.shape[0])]
test_df             = train_test[:int(.20 * train_test.shape[0])]

train               = train_df.to_numpy()
test                = test_df.to_numpy()

kmeans = KMeans(n_clusters=3).fit(train)
results = kmeans.predict(test)