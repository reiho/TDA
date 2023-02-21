# Import the class
from itertools import combinations
import kmapper as km
import matplotlib as mpl
import networkx as nx
import networkx.algorithms.community as nx_comm
import scipy
import statsmodels.api as sm
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import json
from os.path import exists
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

class clustering: 
    def __init__(self, affinity, t):
        self.affinity=affinity
        self.t=t
    def get_params(self):
        return {'affinity': self.affinity,
                'compute_distances': False,
                'compute_full_tree': 'auto',
                'connectivity': None,
                'distance_threshold': None,
                'linkage': 'ward',
                'memory': None,
                'n_clusters': 2}
    def fit(X):
        return None
    
    def predict(self, X):
        Xd=distance(self.affinity, X)
        Z=scipy.cluster.hierarchy.single(Xd)
        return scipy.cluster.hierarchy.fcluster(Z, t=self.t, criterion='distance')
    def fit_predict(self, X):
        Xd=distance(self.affinity, X)
        Z=scipy.cluster.hierarchy.single(Xd)
        return scipy.cluster.hierarchy.fcluster(Z, t=self.t, criterion='distance')

def stat_test(df, df0, columns, test_number):
    results=pd.DataFrame(columns=['gene', 'pvalue', 'mean', 'mean0'])
    print('Performing statistical tests')
    bar=tqdm(total=len(columns))
    for col in columns:
        bar.update(1)
        try:
            pv=scipy.stats.mannwhitneyu(df[col], df0[col]).pvalue
        except:
            pv=1
        try:
            pv2=scipy.stats.fisher_exact(pd.crosstab(np.concatenate([df[col], 
                                                                     df0[col]]), 
                                                     np.concatenate([[1]*df.shape[0], 
                                                                     [0]*df0.shape[0]]))).pvalue
        except:
            pv2=scipy.stats.chi2_contingency(pd.crosstab(np.concatenate([df[col], 
                                                                         df0[col]]), 
                                                         np.concatenate([[1]*df.shape[0], 
                                                                         [0]*df0.shape[0]])))[1]

        if min([pv, pv2])<0.05/len(columns)/test_number:
            results=pd.concat([results, pd.DataFrame({'gene':col, 
                                                   'pvalue':min([pv, pv2]),
                                                   'pvalue_adj': min([pv, pv2])*len(columns)*test_number,
                                                   'mean':df[col].quantile(0.5), 
                                                   'mean0':df0[col].quantile(0.5)}, index=[0])], 
                              axis=0, ignore_index=True)

    return results

def clusterization_results(clusters_nodes, graph, G_s, X, filter_f, folder_name, KM, plotplots=False, PH=False):

    clusters=clusteres_dict_maker(clusters_nodes, graph)

    all_pairs_dict={}

    null=X.replace(0, np.nan).min().min()/10
    
    no_clust=False
    #print(clusters_nodes)
    clusters_nodes=np.array(clusters_nodes)
    if clusters_nodes.shape[0]==1: no_clust=True
    
    for k1, k2 in combinations(clusters.keys(), 2):
        if no_clust:             
            path=[clusters[n]['nodes'][0] for n in list(clusters.keys()) if n!=k1 and n!=k2]
            #print(path)
        try:
            path = nx.shortest_path(G_s, list(clusters[k1]['nodes'])[0], list(clusters[k2]['nodes'])[0])
        except:
            path=[]

        df=X.iloc[clusters[k1]['people']]
        df0=X.iloc[clusters[k2]['people']]
        #display(X)
        res=stat_test(df, df0, X.columns, len(list(combinations(clusters.keys(), 2))))
        if res.shape[0]==0: 
            continue

        print('--------------------------')
        print(k1, k2)
        print(clusters[k1]['nodes'])
        print(clusters[k2]['nodes'])
        print(path)
        print('--------------------------')

        
        dict0=res.set_index('gene').drop(['mean', 'mean0'], axis=1).T.to_dict()


        path_medians=np.array([res.set_index('gene')['mean'].values])
        path_full={}
        for g in res.gene:
            dict0[g]['path']={}
        for g in res.gene:
            dict0[g]['path'][k1]=list(np.log(df[g].replace(0, null).tolist()))

        for n in path:
            if n not in clusters[k1]['nodes'] and n not in clusters[k2]['nodes']:
                #print(graph['nodes'][n])
                #display(X.iloc[graph['nodes'][n]])
                #print(X.iloc[graph['nodes'][n]][res.gene.values])
                path_medians=np.concatenate([path_medians,
                                            np.array([X.iloc[graph['nodes'][n]][res.gene.values].quantile(0.5).T.to_numpy()])])
                for g in res.gene:
                    dict0[g]['path'][n]=list(np.log(X.iloc[graph['nodes'][n]][g].replace(0, null).to_numpy()))
        path_medians=np.concatenate([path_medians,
                                     np.array([res.set_index('gene')['mean0'].T.to_numpy()])])
        #print(path_medians)
        for g in res.gene:
            dict0[g]['path'][k2]=list(np.log(df0[g].replace(0, null).tolist()))

        path_medians=np.transpose(path_medians)

        for (g, m) in zip(res.gene, path_medians):
            dict0[g]['path_medians']=list(m)

        if plotplots:
            for (g, pv) in res[['gene', 'pvalue_adj']].values:
                mypal={}
                for k, name in enumerate(dict0[g]['path'].keys()):
                    if 'cube' in str(name):
                        mypal[k]="#2D46B9"
                    else:
                        mypal[k]="#00FFAB"
                #path_length=len(dict0[g]['path'].values())
                #colors=np.array(["#2D46B9"]*path_length)
                #colors[-1]="#F037A5"
                #colors[0]="#F037A5"
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.boxplot(data=list(dict0[g]['path'].values()), palette=mypal)
                sns.stripplot(data=list(dict0[g]['path'].values()), size=2, color='gray', alpha=0.6)
                ax.set_xticklabels(list(dict0[g]['path'].keys()), rotation=45)
                plt.title(g+' p-value='+str("{:e}".format(pv)))
                if KM: plt.savefig(folder_name+'KMresult'+filter_f+str(g)+'.png', dpi=300)
                else: plt.savefig(folder_name+'result'+filter_f+str(g)+'.png', dpi=300)
                #plt.show()
                plt.cla()

        all_pairs_dict[str(k1)+'vs'+str(k2)]=dict0
    #print(all_pairs_dict)
    if KM: filename=folder_name+'KMresult'+filter_f+'.json'
    elif not PH: filename=folder_name+'result'+filter_f+'.json'
    elif PH:
        if len(all_pairs_dict)==0: return
        filename = folder_name + 'cycles_result' + filter_f + '.json'
        i=0
        while True:
            file_exists = exists(filename)
            if not file_exists:
                continue
            else:
                i=i+1
                filename = folder_name + 'cycles_result' + filter_f +str(i)+ '.json'
    print('writing results to', filename)
    with open(filename, 'w') as outfile:
        json.dump(all_pairs_dict, outfile)


def graph_stat_clustering(number_of_clusters, G, th, deg_cen, filter_f, folder_name):
    centers = deg_cen[deg_cen > th].index.to_list()
    G_c = G.subgraph(centers)
    #nx.draw(G)
    #plt.show()
    clusters_nodes = list(nx.connected_components(G_c))

    bet_cen = pd.Series(nx.betweenness_centrality(G))
    plt.hist(list(bet_cen.values), bins=40)
    plt.title('Betweenness Centrality distribution')
    plt.savefig(folder_name + 'BetweennessCentrality'+filter_f+'.png', dpi=300)
    plt.cla()
    for perc in reversed(range(100)):

        th = bet_cen.sort_values().iloc[int(len(bet_cen) * perc / 100)]
        # th=1
        briges = deg_cen[bet_cen > th].index.to_list()
        G_s = G.subgraph(briges + centers)
        nb = len(list(nx.connected_components(G_s)))
        if nb == 1:
            break

    print('Between centrality threshold', th)
    f = plt.figure(figsize=(15, 15))
    nx.draw_spring(G_s,
                   node_color=['#00FFAB' if n in centers else '#2D46B9' for n in G_s.nodes()],
                   with_labels=False)
    plt.title('The subgraph with cluster centers and paths inbetween')
    plt.savefig(folder_name + 'clusters_subgraph_'+filter_f+'.png', dpi=300)
    plt.show()

    if nb != 1: return None, np.array(clusters_nodes)
    return G_s, np.array(clusters_nodes)

def network_analysis_with_given_parameters(n_cubes, perc_overlap, X, target, t0, filter_f, folder_name,
                                           number_of_clusters_only=False, plotplots=True):
    # Initialize
    mapper = km.KeplerMapper(verbose=1)

    # Create the simplicial complex
    #print(int(n_cubes), perc_overlap)
    #print(X.shape, target.shape, t0)
    #print(X)
    graph = mapper.map(
        target,
        X,
        precomputed=False,
        cover=km.Cover(n_cubes=int(n_cubes), perc_overlap=perc_overlap),
        clusterer=clustering(affinity='spearman', t=t0),
    )

    if len(graph['nodes']) == 0 or len(graph['links']) == 0:
        print('Not possible to build a graph')
        #print(graph)
        if number_of_clusters_only:
            return 0, graph, None

    G = nx.Graph()
    G.add_nodes_from(graph['nodes'].keys())
    for l in graph['links']:
        for l2 in graph['links'][l]:
            G.add_edge(l, l2)


    
    deg_cen = pd.Series(nx.degree_centrality(G))
    #print(deg_cen)
    if not number_of_clusters_only: 
        #nx.draw_spring(G)
        #plt.show()
        #print(list(deg_cen.values))
        plt.hist(list(deg_cen.values), bins=20)
        plt.title('Degree Centrality distribution')
        plt.show()
    
    percs=[]
    ncs=[]
    for perc in range(100):
        percs.append(perc)
        #print(int(len(deg_cen)*perc/100))
        #display(deg_cen.sort_values())
        th=deg_cen.sort_values().iloc[int(len(deg_cen)*perc/100)]
        
        centers=deg_cen[deg_cen>th].index.to_list()
        G_c=G.subgraph(centers)
        #nx.draw_spring(G_c, with_labels=True)
        nc=len(list(nx.connected_components(G_c)))
        ncs.append(nc)
    if not number_of_clusters_only: 
        plt.plot(percs, ncs)
        plt.title('Degree сentrality')
        plt.xlabel('quantile for degree centrality distribution')
        plt.ylabel('number of connected components (clusters)')
        plt.savefig(folder_name+'DegreeCentrality'+filter_f+'.png', dpi=300)
        plt.show()
    number_of_clusters=pd.Series(ncs).max()
    print(number_of_clusters_only)

    if number_of_clusters_only: 
        return number_of_clusters, graph, G
    if number_of_clusters>1:
        perc=np.array(percs)[ncs==number_of_clusters].max()
        th=deg_cen.sort_values().iloc[int(len(deg_cen)*perc/100)]
        print('Degree centrality quantile', th, perc, '%')
        G_s, clusters_nodes = graph_stat_clustering(number_of_clusters, G, th, deg_cen, filter_f, folder_name)
        clusterization_results(clusters_nodes, graph, G_s, X, filter_f, folder_name,  KM=False, plotplots=plotplots)

        clusters_nodes=KMeans_graph_clustering(G, number_of_clusters, filter_f, folder_name)
        clusterization_results(clusters_nodes, graph, G, X, filter_f, folder_name, KM=True, plotplots=plotplots)
    else: print('clusteres not found' )


def KMeans_graph_clustering(G, number_of_clusters, filter_f, folder_name):
    L = nx.laplacian_matrix(G).astype(float)
    w,v = scipy.sparse.linalg.eigsh(L, k = int(L.shape[0]*0.8)
                                    , which='SM')

    x = v*w
    print(number_of_clusters)
    #print(x)
    kmeans = KMeans(init='k-means++', n_clusters=number_of_clusters, n_init=50)
    kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    error = kmeans.inertia_
    
    #base = plt.cm.get_cmap('viridis')
   # colors = base(np.linspace(0, 1, number_of_clusters))
    node_colors = [ colors[labels[v]] for v, n in enumerate(G.nodes())]
    f=plt.figure(figsize=(15,15))
    nx.draw(G, node_color=node_colors, with_labels=False)
    plt.title('KMeans graph clustering results')
    plt.savefig(folder_name + '/KMeans_clusters' + filter_f + '.png', dpi=300)
    plt.show()
    
    clusters=[]
    for label in np.unique(labels):
        clusters.append(set(np.array(G.nodes())[labels==label]))
    
    return clusters


def clusteres_dict_maker(clusters_nodes, graph):
    clusters={}
    for j, cl in enumerate(clusters_nodes):
        i='cluster '+str(j)
        clusters[i]={}
        clusters[i]['nodes']=cl
        people=[]
        for n in cl:
            people=people+graph['nodes'][n]
        people=np.unique(people)
        clusters[i]['people']=people
    return clusters

def n_clust_in_graph(G):
    percs=[]
    ncs=[]
    for perc in range(100):
        percs.append(perc)
        th=deg_cen.sort_values().iloc[int(len(deg_cen)*perc/100)]
        centers=deg_cen[deg_cen>th].index.to_list()
        G_c=G.subgraph(centers)
        #nx.draw_spring(G_c, with_labels=True)
        nc=len(list(nx.connected_components(G_c)))
        ncs.append(nc)
    number_of_clusters=pd.Series(ncs).value_counts().idxmax()
    return number_of_clusters


def mapper_modularity(X, n_cubes,  p_o, target, t0, filter_f, folder_name):
        # Create the simplicial complex
    perc_overlap=p_o/100
    print(X.shape)
    #try:
    number_of_clusters, graph, G = network_analysis_with_given_parameters(n_cubes, perc_overlap, X, target, t0,
                                                                          filter_f, folder_name,
                                                                           number_of_clusters_only=True)
    #except:
    #    print('Failed', n_cubes,  p_o)
    #    return 0, 0   
    #print(graph['nodes'])
    if len(graph['nodes'])<2 or G==None:
        return 0, 0, 0
        
    comm=nx.algorithms.community.centrality.girvan_newman(G)
    comm=list(comm)
    return nx_comm.modularity(G, list(comm[0])), number_of_clusters, len(list(nx.connected_components(G)))

def lucas(n):
    # Base cases
    if n == 0:
        return 2
    if n == 1:
        return 1
   
    # recurrence relation
    return lucas(n - 1) + lucas(n - 2);

def iterative_mapper_fibonacci(target, X, t0, filter_f, folder_name,
                     n_cubes_l=1, n_cubes_u=20,overlap_l=10, overlap_u=90,iterations=5):
    z=[]

    #cubes
    xmin=n_cubes_l
    xmax=n_cubes_u
    x=[]

    #overlaps
    ymin=overlap_l
    ymax=overlap_u
    y=[]
    
    #number of clusters
    nc=[]
    #number of connected components
    ncoms=[]

    bar=tqdm(total=iterations)

    L=[]
    for it in range(1, iterations+3):
        L.append(lucas(it))
    L=np.array(L)

    pace=[]

    N=iterations
    for it in range(1, iterations):
        bar.update(1)
        x1=int(xmin+(xmax-xmin)*L[N-it]/L[N-it+2])
        y1=ymin+(ymax-ymin)*L[(N-it)]/L[(N-it+2)]
        #print('y1', y1, L[(N-it)]/L[(N-it+2)])
        x2=int(xmin+(xmax-xmin)*L[(N-it+1)]/L[(N-it+2)])
        y2=ymin+(ymax-ymin)*L[(N-it+1)]/L[(N-it+2)]

        mod_max=0
        new_points=np.array([[xmin,  y1], [x1,  ymin], [x1,  y1], 
                       [x2,  ymax], [xmax,  y2], [x2,  y2]])
        c_new=None
        for i, [c0, p0] in enumerate(new_points):
            exists=False
            for c00, p00, z00 in zip(x, y, z):
                if c0==c00 and p0==p00: 
                    exists=True
                    mod=z00
            if not exists:
                x.append(c0)
                y.append(p0)

                mod, nclus, ncom = mapper_modularity(X, c0, p0, target, t0, filter_f, folder_name)
                z.append(mod)
                nc.append(nclus)
                ncoms.append(ncom)
            if mod!=mod: continue
            if mod>mod_max:
                mod_max=mod
                if i<3: 
                    min_flag=True

                    c_new, p_new = new_points[i+3]

                else:
                    min_flag=False

                    c_new, p_new = new_points[i-3]

        #print(c_new, p_new, mod_max)
        if c_new!=None:
            pace.append([c_new, p_new, mod_max])
            if min_flag:
                xmax=c_new
                ymax=p_new
            else:
                xmin=c_new
                ymin=p_new

    z=np.array(z)
    x=np.array(x)
    y=np.array(y)
    nc=np.array(nc)
    ncoms=np.array(ncoms)
    
    plot_diagrams(x, y, z, nc, filter_f, folder_name)
    
    return x, y, z, nc, ncoms

def plot_diagrams(x, y, z, nc, filter_f, folder_name):
    fig, ax = plt.subplots(2,2, figsize=(6,6), gridspec_kw={'height_ratios': [9, 1]})

    cmap = mpl.cm.cool
    ax[0][0].scatter(x, y, c=z, s=200, alpha=0.7, cmap=cmap)
    ax[0][0].set_xlabel('number of cubes')
    ax[0][0].set_ylabel('overlape')

    norm = mpl.colors.Normalize(vmin=min(z[z==z]), vmax=max(z[z==z]))
    cb1 = mpl.colorbar.ColorbarBase(ax[1][0],
                                    norm=norm,
                                    orientation='horizontal', cmap=cmap)
    cb1.set_label('modularity')

    cmap = mpl.cm.spring_r
    ax[0][1].scatter(x, y, c=nc, s=200, alpha=0.7, cmap=cmap)
    ax[0][1].set_xlabel('number of cubes')
    ax[0][1].set_ylabel('overlape')

    norm = mpl.colors.Normalize(vmin=min(nc[nc==nc]), vmax=max(nc[nc==nc]))
    cb1 = mpl.colorbar.ColorbarBase(ax[1][1],
                                    norm=norm,
                                    orientation='horizontal', cmap=cmap)
    cb1.set_label('number of clusters')
    plt.savefig(folder_name+'/fibonacci_search'+filter_f+'.png', dpi=300)
    plt.show()
