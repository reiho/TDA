
import gudhi.representations
import numpy as np
import pandas as pd
import kmapper as km
from biotda.mapper_pipline import clusterization_results, clustering, distance
import networkx as nx
from tqdm import tqdm_notebook as tqdm
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
import cv2
import matplotlib as mpl
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


def cycles(n_cubes, perc_overlap, X, target, t0, filter_f, folder_name, test_only_most_variable_gens):

    mapper = km.KeplerMapper(verbose=1)

    # Create the simplicial complex
    graph = mapper.map(
        target,
        X,
        precomputed=False,
        cover=km.Cover(n_cubes=int(n_cubes), perc_overlap=perc_overlap),
        clusterer=clustering(affinity='spearman', t=t0),
    )

    G = nx.Graph()
    G.add_nodes_from(graph['nodes'].keys())
    for l in graph['links']:
        for l2 in graph['links'][l]:
            G.add_edge(l, l2)

    graphDM=pd.DataFrame(index=graph['nodes'].keys(), columns=graph['nodes'].keys())
    for i, n1 in enumerate(graph['nodes'].keys()):
        for n2 in list(graph['nodes'].keys())[i+1:]:
            node1=graph['nodes'][n1]
            node2=graph['nodes'][n2] 
            v=len(set(node1) & set(node2))/len(set(node1) | set(node2))
            #print(v)
            graphDM.loc[n1, n2]=v
            graphDM.loc[n2, n1]=v  
    graphDM=graphDM.fillna(1)

    max_dimension=1
    max_edge_length=int(graphDM.max().values[0]*100)
    graphDM=(1-graphDM)

    rips = gudhi.RipsComplex(distance_matrix=graphDM.to_numpy()*100,
                             max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_dimension)
    #VR = VietorisRipsPersistence(homology_dimensions=[1], 
    #                         metric='precomputed')  # Parameter explained in the text
    #diagrams = VR.fit_transform(distance_matrix=(1-graphDM).to_numpy()*100)
    #print(rips.persistence())
    for i, (lb, ub) in rips.persistence(persistence_dim_max=True):
        if i!=1:
            print('Something wrong', i)
            break
        cycle_nodes=graphDM[(graphDM>=lb/100)&(graphDM<=ub/100)].dropna(how='all').index
        if len(cycle_nodes)<3:
            print('Cycle is too small', len(cycle_nodes))
            break
        G_cycle=G.subgraph(cycle_nodes)
        #try:
        if True:
            cycle_edges=nx.find_cycle(G_cycle)
            G_cycle=G_cycle.edge_subgraph(cycle_edges)

            f=plt.figure(figsize=(6,6))
            nx.draw_spring(G_cycle,
                           node_color=['#F037A5']*len(G_cycle.nodes),
                          with_labels=True, alpha=0.5)
            plt.savefig(folder_name+'cycle_'+str(lb)+'_'+str(ub)+filter_f+'.png', dpi=300)
            #plt.show()
            plt.close(f)
            #print(G_cycle.nodes)
            medians=pd.DataFrame(columns=G_cycle.nodes, index=X.columns)

            for node in G_cycle.nodes:
                medians[node]=X.iloc[graph['nodes'][node]].quantile(0.5).values

            most_var=(medians.var(axis=1)/medians.max(axis=1).replace(0, 1)).sort_values()[-10:].index
            medians.to_csv(folder_name+'cycle_medians_'+str(lb)+'_'+str(ub)+filter_f+'.csv')
            #print('Statistic tests')
            #plotplots=False
            #if test_only_most_variable_gens:
            #    clusterization_results(np.transpose(np.array([G_cycle.nodes])), graph, G_cycle, X[most_var],
            #                           filter_f, folder_name,  PH=True, plotplots=plotplots, KM=False,)
            #else:
            #    clusterization_results(np.transpose(np.array([G_cycle.nodes])), graph, G_cycle, X,
            #                          filter_f, folder_name,  PH=True, plotplots=plotplots, KM=False,)


def PL(X, p, folder_name, sign):
    if len(p)>5000:
        most_var = list(((X.var()-X.min()) / (X.max()-X.min())).sort_values()[-4000:].index)
        p=np.unique(most_var+sign)

    X=X[p].T.to_numpy()
    #print(X.shape)
    M = 1 - distance(affinity='spearman', X=X)

    XPL = []
    #print(min(len(p), 10))
    minres=min(len(p), 10)
    maxres=int(X.shape[0]/5)
    maxres=max(minres, maxres)
    for tmax in range(minres, maxres, 10):
        #print(tmax)
        change=False
        XPL = []
        bar = tqdm(total=X.shape[1])
        for k in range(X.shape[1]):
            bar.update(1)
            t = M.copy()
            for i in range(len(p)):
                t[i] = (M[i, :] ** 2 + ((X[i, k] ** 2 - X[:, k] ** 2) / M[i, :]) ** 2 + 2 * X[i, k] ** 2 + 2 * X[:,
                                                                                                               k] ** 2) ** 0.5 / 2
            # giotto
            VR = VietorisRipsPersistence(homology_dimensions=[1],
                                         metric='precomputed')  # Parameter explained in the text
            # d=int(t.shape[0]/2)
            d = t.shape[0]

            diagrams = VR.fit_transform(np.nan_to_num(t[:d, :d]).reshape(1, d, d))
            LS = PersistenceLandscape(n_layers=max([2, int(tmax / 4)]),
                                      n_bins=tmax)
            L = LS.fit_transform(diagrams)

            LI = np.flip(np.reshape(L[0], [max([2, int(tmax / 4)]), tmax]), 0)

            LI = LI[np.sum(LI, axis=1) > 0]
            XPL.append(LI)

            if np.sum(LI) == 0:
                change=True
                print('changing resolution to', tmax+10)
                continue

        if not change: break

    d1 = XPL[0].shape[1]
    d2 = max([p.shape[0] for p in XPL])

    PL0 = np.array([cv2.resize(p, (d1, d2)) if p.shape[0] > 0 else np.array([np.array([0] * d1)] * d2) for p in XPL])

    f = plt.figure(figsize=(8, 8))
    plt.imshow(PL0[0], aspect='auto', cmap=mpl.cm.cividis)
    plt.grid(False)
    plt.title("Persistence Landscape - the first sample")
    plt.show()

    np.array(PL0).dump(folder_name + 'PLandscape_full.npy')
