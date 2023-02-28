# Import the class
import numpy as np
import pandas as pd
import random
from itertools import combinations
import networkx as nx
import skbio
import sklearn
import statsmodels.api as sm
from combat.pycombat import pycombat
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')

plt.rcParams['image.cmap'] = 'icefire'

global colors
colors=["#14C38E","#F037A5","#2D46B9", '#E8FFC2', '#00FFAB', '#2FA4FF',
        '#C400FF','#78DEC7',  '#7900FF',  '#FFF338','#400082', '#FF5403',  '#FFCA03',
        '#0D7377', '#D2E603']
sns.set_palette(sns.color_palette(colors,16))

def mylr(X, Y):
    X0=X.to_numpy()
    #print(X0, Y)
    X0 = sm.add_constant(X0)
    #scaler = preprocessing.StandardScaler().fit(X0)
    #X0 = scaler.transform(X0)
    #print(X0, Y)
    lr=sm.OLS(Y, X0)
    results = lr.fit()
    return X0, results

def distance(affinity, X):
    if affinity=='spearman':
        Xd=pd.DataFrame(X).astype('float').T.corr(method='spearman').values
    elif affinity=='pearson':
        Xd=pd.DataFrame(X).astype('float').T.corr(method='pearson').values
    else:
        print('Unsuppurted affinity. Available methods: "spearman", "pearson"')
    return Xd

def check_for_batch_effect(d, p, covariates_columns, target_column, plot):
    ps=[]
    bs=[]
    vs=[]
    for p0 in p:
        X=d[np.concatenate([[p0], covariates_columns])].astype('float').dropna()
        X0, res=mylr(X, d.loc[X.index][target_column])
        if res.pvalues[1]<0.05/len(p):
            ps.append([p0]*d.shape[0])
            bs.append(d[target_column])
            vs.append(d[p0].values)


    ps=np.ravel(np.array(ps))
    bs=np.ravel(np.array(bs))
    vs=np.ravel(np.array(vs))
    print('Number of components', len(np.unique(ps)))
    
    if plot and len(np.unique(ps))>0:
        f=plt.figure(figsize=(10,15))
        sns.boxplot(y=ps, x=vs, hue=bs, color='#E8FFC2')
        plt.show()
        
    return len(np.unique(ps))

def batch_effect(d, covariates_columns, p, target_column, plot=False):
    
    nc=check_for_batch_effect(d, p, covariates_columns, target_column, plot)
    
    if nc>0:
        print('Reducing batch effect')
        cov=d[covariates_columns].dropna().astype('int')
        d_new=d.copy()
        d_new.loc[cov.index, p]=pycombat(d.loc[cov.index,p].T.astype('float'), 
                 d.study.loc[cov.index].values,
                 mod=cov.T.values.tolist()
                ).T

    
    nc=check_for_batch_effect(d_new, p, covariates_columns, target_column, plot)
    return(d_new)


def DSGA_filter(d_merged, D, N, p,n_components=2):
    print('Calculating DSGA filter')
    d_merged=d_merged[p].dropna()
    D=np.array([d for d in D if d in d_merged.index])
    #
    N=np.array([d for d in N if d in d_merged.index])
    
    #print(len(D), len(N))
    pca=PCA(n_components=n_components)
    p_pca=pca.fit_transform(StandardScaler().fit_transform(d_merged.loc[N, p]))
    print('PCA for control, explained variance ratio', np.sum(pca.explained_variance_ratio_))
    
    d_flat=pd.DataFrame(columns=p, index=N)
    bar=tqdm(total=len(N))
    for i in N:
        bar.update(1)
        #print(list(N[N!=i]))
        #print(min(len(p)-1, len(N)-1))
        X=d_merged.loc[random.sample(list(N[N!=i]), k=min(len(p)-1, len(N)-1)), p].T.astype('float')
        Y=d_merged.loc[i, p].astype('float').values
        #display(X)
        X0=X.to_numpy()
        #X0 = sm.add_constant(X0)

        lr=sm.OLS(Y, X0)
        res = lr.fit()

        d_flat.loc[i]=res.predict(X0)
        
    pca=PCA()
    p_pca=np.transpose(pca.fit_transform(d_flat))
    #print(p_pca.shape)
    print('PCA for flat control, explained variance ratio', 
          np.sum(pca.explained_variance_ratio_[:n_components]))
    
    d_flatD=pd.DataFrame(columns=p, index=D)
    d_residD=pd.DataFrame(columns=p, index=D)
    bar=tqdm(total=len(D))
    for i in D:
        bar.update(1)
        X=d_merged.loc[random.sample(list(N), k=min(len(p)-1, len(N)-1)), p].T.astype('float')
        Y=d_merged.loc[i, p].astype('float').values
        X0, res=mylr(X, Y)
        #print(X0)
        d_flatD.loc[i]=res.predict(X0)
        d_residD.loc[i]=res.resid
        

    P=np.dot(np.dot(p_pca, np.linalg.inv(np.dot(np.transpose(p_pca), p_pca))), np.transpose(p_pca))

    Nc=np.dot(pca.transform(d_flatD), P)
    Dc=pca.transform(d_flatD)-Nc
    Nc=np.transpose(p_pca)
    target=np.concatenate([Dc[:, :2], Nc[:, :2]])
    return target

def fpk(data, p=2, k=1):
    return ((data**p).sum(axis=1))**(k/p)


def KNearest_graph(d_merged, p, D, N, n_s1=None, n_s2=None):
    data=d_merged.loc[np.concatenate([D[:n_s1], N[:n_s2]]),p].dropna()
    knn=sklearn.neighbors.kneighbors_graph(data,
                                           n_neighbors=min([int((len(D)+len(N))/20),30]), mode='distance')
    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)
    adj={}
    for u, row in enumerate(knn.toarray()):
        for v, w in enumerate(row):
            if w!=0:
                if u in adj.keys():
                    arr=adj[u]
                    arr.append((v, w))
                    adj[u]=arr
                else:
                    adj[u]=[(v, w)]
                    
    G = nx.Graph()
    G.add_nodes_from(adj.keys())
    for l in adj.keys():
        for l2 in adj[l]:
            G.add_edge(l, l2[0], weight=l2[1])

    dist=np.matrix([[0]*len(list(adj.keys()))]*len(list(adj.keys())))
    print('Computing paths from KNearest graph')
    bar=tqdm(total=len(list(combinations(list(range(dist.shape[0])), 2))))
    for i, j in combinations(list(range(dist.shape[0])), 2):  
        bar.update(1)
        try:
            dist[i, j]=nx.shortest_path_length(G, source=i, target=j)
        except:
            print('fail')
            #print(dist)
            dist[i, j]=-1
        dist[j, i]=dist[i, j]
    return dist

def KNearest_filter(d_merged, D, N, p, folder_name, n_s1=None, n_s2=None, knn_dist_path=None):
    print('Calculating KNearest filter')
    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)
    if knn_dist_path!=None:
        dist=pd.read_csv(knn_dist_path, header=None).to_numpy()
        print(dist.shape)
    else:
        dist=KNearest_graph(d_merged, p, D, N, n_s1=len(D), n_s2=len(N))
        np.savetxt(folder_name+'knn_distance.csv', dist, delimiter=',')

        
    pcoa=skbio.stats.ordination.pcoa(dist, number_of_dimensions=2)
    return pcoa.samples.values

def MDS_filter(d_merged, p, cov_col, D, N, n_s1=None, n_s2=None):
    print('Calculating MDS filter')
    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)
    dist=distance('spearman',
                  d_merged.loc[np.concatenate([D[:n_s1], N[:n_s2]]),p+cov_col].dropna())
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(dist)
    return X_transformed

def PCA_filter(d_merged, p, cov_col, D, N, n_s1=None, n_s2=None):
    print('Calculating PCA filter')
    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)
    embedding = PCA(n_components=2)
    X_transformed = embedding.fit_transform(d_merged.loc[np.concatenate([D[:n_s1], N[:n_s2]]),p+cov_col].dropna())
    return X_transformed

def PCAflat_filter(d_merged, p, cov_col, D0, N0, n_s1=None, n_s2=None, n_components=2):
    print('Calculating flat PCA filter')
    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)
    # print(len(D), len(N))

    N=np.concatenate([D0[:n_s1],N0[:n_s2]])
    d_flat = pd.DataFrame(columns=p, index=N)
    bar = tqdm(total=len(N))
    for i in N:
        bar.update(1)
        X = d_merged.loc[random.sample(list(N[N != i]), k=min(len(p) - 1, len(N) - 1)), p].T.astype('float')
        Y = d_merged.loc[i, p].astype('float').values
        X0 = X.to_numpy()
        lr = sm.OLS(Y, X0)
        res = lr.fit()
        d_flat.loc[i] = res.predict(X0)
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(d_flat)
    return X_transformed

