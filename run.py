import pandas as pd
import mapper_pipline
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy
import PH_pipline


def run(d_merged, D, N, p, covariates_columns,
        folder_name, fix_batch_effect=False, batch_column=None,
        filters=['DSGA',
            'KNearest',
            'MDS',
            'PCA',
            'PCAflat'],
        knn_dist_path=None,
        n_s1=None,
        n_s2=None,
        overlap_l=10, num_groups=0,
        PH=False, test_only_most_variable_gens=True, sign=[]):
    if fix_batch_effect:
        d_merged_new=preprocessing.batch_effect(d_merged, covariates_columns, p, batch_column)

    if n_s1==None: n_s1 = len(D)
    if n_s2==None: n_s2 = len(N)

    X = pd.concat([d_merged.loc[D[:n_s1], covariates_columns + p],
                       d_merged.loc[N[:n_s2], covariates_columns + p]]).dropna().astype('float')
    X = X.dropna()

    for filter_f in filters:
        try:
            with open(folder_name + filter_f + 'target.npy', 'rb') as f:
                target = np.load(f)
        except:
            if filter_f == 'DSGA':
                target = preprocessing.DSGA_filter(d_merged, D, N, covariates_columns + p)
            elif filter_f == 'KNearest':
                target = preprocessing.KNearest_filter(d_merged, D, N, covariates_columns + p, folder_name, n_s1, n_s2,
                                                       knn_dist_path=knn_dist_path)
            elif filter_f == 'MDS':
                target = preprocessing.MDS_filter(d_merged, p, covariates_columns, D, N, n_s1, n_s2)
            elif filter_f == 'PCA':
                target = preprocessing.PCA_filter(d_merged, p, covariates_columns, D, N, n_s1, n_s2)
            elif filter_f == 'PCAflat':
                target = preprocessing.PCAflat_filter(d_merged, p, covariates_columns, D, N, n_s1, n_s2)
            with open(folder_name + filter_f + 'target.npy', 'wb') as f:
                np.save(f, target)

       

        if X.shape[0]<len(target):
            target = target[[d in X.index for d in D] + [d in X.index for d in N]]
        elif X.shape[0]>len(target):
            D=np.array([d for d in D if d in X.index])
            N=np.array([d for d in N if d in X.index])

        try:
            with open(folder_name + '/t0' + filter_f + '.json', 'r') as f:
                output = json.load(f)
                t0 = output['t0']
        except:
            C = scipy.cluster.hierarchy.linkage(X)[:, 2]
            f, ax = plt.subplots(1, 1, figsize=(15, 5))
            sns.histplot(C, bins=35)
            plt.xlabel('Distance to merge clusters')
            hi, bins = np.histogram(C, bins=35)
            max_h = max(hi)
            max_flag = False
            for h, b in zip(hi, bins):
                if h == max_h:
                    max_flag = True
                if max_flag and h == 0:
                    t0 = b
                    break
            ax.axvline(x=t0, color='#F037A5', alpha=0.8)
            f.patch.set_facecolor('white')
            f.savefig(folder_name + '/clusterization_cutoff_' + filter_f + '.png',
                      dpi=300, facecolor=f.get_facecolor())
            plt.show()
            with open(folder_name + '/t0' + filter_f + '.json', 'w') as outfile:
                json.dump({'t0': t0}, outfile)


        try:
            with open(folder_name + 'params' + filter_f + '.json', 'r') as f:
                output = json.load(f)
                n_cubes = output['n_cubes']
                perc_overlap = output['overlap']

        except:
            x, y, z, nc, ncoms = mapper_pipline.iterative_mapper_fibonacci(target, X, t0,
                                                                    filter_f, folder_name,
                                                                    n_cubes_l=3, n_cubes_u=20,
                                                                    overlap_l=overlap_l, overlap_u=70,
                                                                    iterations=20)

            if nc.max() == 1:
                x, y, z, nc, ncoms = mapper_pipline.iterative_mapper_fibonacci(target, X, t0,
                                                                        filter_f, folder_name,
                                                                        n_cubes_l=3, n_cubes_u=25,
                                                                        overlap_l=5,
                                                                        overlap_u=90, iterations=25)
                if nc.max() == 1:
                    print('No optimal parameters for Mapper')
                    nc = np.array([2] * len(z))
            #print(z, nc, ncoms)
            sel=z[((nc > 1) & (ncoms<=min(ncoms)+num_groups))]
            while len(sel)==0:
                num_groups=num_groups+1
                sel=z[((nc > 1) & (ncoms<=min(ncoms)+num_groups))]
            n_cubes = int(x[z == max(sel)][0])
            perc_overlap = y[z == max(sel)][0] / 100
            print('------------------')
            print('Selected n_cubes:', n_cubes, '; overlap:', perc_overlap)
            print('Filter function is', filter_f)
            print('------------------')

            with open(folder_name + 'params' + filter_f + '.json', 'w') as outfile:
                json.dump({'n_cubes': n_cubes, 'overlap': perc_overlap}, outfile)

        if not PH: mapper_pipline.network_analysis_with_given_parameters(n_cubes, perc_overlap, X, target, t0,
                                                              filter_f, folder_name)

        print(filter_f)
        if PH:
            print('PH pipline')
            PH_pipline.cycles(n_cubes, perc_overlap, X, target, t0, filter_f, folder_name, test_only_most_variable_gens)
    if PH:
        print(p)
        X = pd.concat([d_merged.loc[D[:n_s1], p],
                       d_merged.loc[N[:n_s2], p]]).dropna().astype('float')
        PH_pipline.PL(X, p, folder_name, sign)
