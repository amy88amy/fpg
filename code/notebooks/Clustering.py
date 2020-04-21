#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import hstack
from gensim.models import Word2Vec
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


data_path = 'bs140513_032310.csv'
raw_data = pd.read_csv(data_path)

# from `bank_sim_dat_exp.ipynb` previous analysis 
def cat_amount(v, mean, median):
    res = ""
    if v > mean:
        res = "above_mean"
    elif v < median:
        res = "below_median"
    elif v >= median and v <= mean:
        res = "in_between"
    return res

amount_data = raw_data["amount"]
mean_amount = amount_data.mean()
median_amount = amount_data.median()
raw_data["amount_cat"] = np.vectorize(cat_amount)(raw_data["amount"].values, mean_amount, median_amount)

pre_data = raw_data[["step", "customer", "age", "gender", "merchant", "category", "amount_cat", "fraud"]]
fraud_data = pre_data[pre_data["fraud"] == 1]
non_fraud_data = pre_data[pre_data["fraud"] == 0]

feat_cols = fraud_data.columns
print("List of feature columns used: {}".format(feat_cols))

f_train, f_test = train_test_split(fraud_data, test_size=0.2)
nf_train, nf_test = train_test_split(non_fraud_data, test_size=0.2)

train_df = pd.concat([f_train, nf_train]).sample(frac = 1)
test_df = pd.concat([f_test, nf_test]).sample(frac = 1)

def get_randomwalk(node, path_length, graph):
    random_walk = [node]
    
    for i in range(path_length-1):
        temp = list(graph.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
        
    return random_walk


# In[4]:


train_G = nx.from_pandas_edgelist(train_df, source="customer", target="merchant", edge_attr=True, create_using=nx.Graph())
test_G = nx.from_pandas_edgelist(test_df, source="customer", target="merchant", edge_attr=True, create_using=nx.Graph())

# train_G = nx.from_pandas_edgelist(train_df, source="customer", target="merchant",\
#                                   edge_attr=True, create_using=nx.DiGraph())
# test_G = nx.from_pandas_edgelist(test_df, source="customer", target="merchant",\
#                                  edge_attr=True, create_using=nx.DiGraph())

train_nodes = list(train_G.nodes())
test_nodes = list(test_G.nodes())

train_walks = []
for n in tqdm(train_nodes):
    for i in range(5):
        train_walks.append(get_randomwalk(n, 10, train_G))
        
test_walks = []
for n in tqdm(test_nodes):
    for i in range(5):
        test_walks.append(get_randomwalk(n, 10, test_G))


# In[5]:


model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(train_walks + test_walks, progress_per=2)
model.train(train_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)


# In[6]:


train_graph_feat = model.wv[train_nodes]
test_graph_feat = model.wv[test_nodes]

train_graph_dict = dict(zip(train_nodes, train_graph_feat))
test_graph_dict = dict(zip(test_nodes, test_graph_feat))


# In[7]:


# Map the nodes with their mass weight
def map_val(row, mass_map):
    return mass_map[row[0]]
# Map all customers/source
# Training data
train_customer_vec = np.apply_along_axis(map_val, 1, train_df["customer"].values.reshape(-1, 1),                                         train_graph_dict)
# Test data
test_customer_vec = np.apply_along_axis(map_val, 1, test_df["customer"].values.reshape(-1, 1),                                        test_graph_dict)
# Map all merchants/targets
# Training data
train_merchant_vec = np.apply_along_axis(map_val, 1, train_df["merchant"].values.reshape(-1, 1),                                         train_graph_dict)
# Test data
test_merchant_vec = np.apply_along_axis(map_val, 1, test_df["merchant"].values.reshape(-1, 1),                                        test_graph_dict)
# Select the required columns
train_data = train_df[["step", "age", "category", "amount_cat", "fraud"]]
test_data = test_df[["step", "age", "category", "amount_cat", "fraud"]]


# In[8]:


cat_cols = ["step", "age", "category", "amount_cat"]

X_train = train_data[["step", "age", "category", "amount_cat"]].values
y_train = train_data["fraud"].values

X_test = test_data[["step", "age", "category", "amount_cat"]].values
y_test = test_data["fraud"].values

X_train_enc = np.array([[None] * len(cat_cols)] * X_train.shape[0])
# Transform categorical columns for training data
label_ens = []
for i in range(0, len(cat_cols)):
    en = LabelEncoder()
    X_train_enc[:, i] = en.fit_transform(X_train[:, i])
    label_ens.insert(i, en)

one_hot_en = OneHotEncoder(handle_unknown='ignore')
X_train_arr = hstack((one_hot_en.fit_transform(X_train_enc[:, 0:len(cat_cols)]),train_customer_vec, train_merchant_vec))

X_test_enc = np.array([[None] * len(cat_cols)] * X_test.shape[0])
# Transform categorical columns for test data
for i in range(0, len(cat_cols)):
    X_test_enc[:, i] = label_ens[i].transform(X_test[:, i])

X_test_arr = hstack((one_hot_en.transform(X_test_enc[:, 0:len(cat_cols)]), test_customer_vec, test_merchant_vec))


# In[9]:


X = X_train_arr.toarray()
X = X[:10000]


# In[10]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50, random_state=0)
kmeans.fit(X)

c = {}
for i,l in enumerate(kmeans.labels_):
    if not l in c:
        c[l] = list()
    c[l].append(X[i])

c


# In[ ]:


Gk = nx.Graph()

fig = plt.gcf()
fig.set_size_inches(10, 10)


# In[11]:


X = X_train_arr.toarray()
X = X[:5000]


# In[14]:


from sklearn.cluster import OPTICS
optics = OPTICS(min_samples=5)
optics.fit(X)
optics.labels_


# In[15]:


# OPTICS
fig = plt.gcf()
fig.set_size_inches(10, 10)

colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[optics.labels_ == klass]
    print(Xk[:, 2], Xk[:, 3])
    plt.plot(Xk[:, 2], Xk[:, 3], color, alpha=0.3)
plt.plot(X[optics.labels_ == -1, 2], X[optics.labels_ == -1, 3], 'k+', alpha=0.1)


# In[ ]:




