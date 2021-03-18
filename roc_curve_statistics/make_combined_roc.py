import sys, os, glob
import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import auc

import numpy as np

rc('font', family = 'serif', serif = 'cmr10')
extension='.eps'
# Get scan info
#save_name='scan_varying'
#all_runs = [('Bone DXA + 2 MRI','SingleDXA.pkl'),
#            ('Tissue DXA + 2 MRI','SingleDXATissue.pkl'),
#            ('Bone DXA + Fat MRI', 'SingleDXASingleMRI.pkl'),
#            ('2 DXA + 2 MRI','BothBoth.pkl'),
#            ('2 DXA + Water MRI','SingleMRIWater3.pkl'),
#            ('2 DXA + Fat MRI','SingleMRI.pkl'),
#            ('Baseline','PooledMaps.pkl')
#            ]


save_name='temperature_varying'
all_runs = [('T=0.1','LowerTemperature1.pkl'),
            ('T=0.05','LowerTemperature05.pkl'),
            ('T=0.01','BothBoth.pkl'),
            ('T=0.005','EqualResLowTemp.pkl'),
            ('T=0.001','LowerTemperature001.pkl'),
            ('Baseline','PooledMaps.pkl')
            ]

title='Contrastive ROC'
roc_fig,  roc_ax =plt.subplots(1,1)
rank_fig, rank_ax=plt.subplots(1,1)

for label, run_save_name in all_runs:
    #label = '.'.join(run_save_name.split('.')[:-1])
    with open(run_save_name,'rb') as f:
        [val_stats, similarities] = pickle.load(f)
    roc_points = []

    # get tpr and fpr
    for threshold in np.linspace(-1,1,2000):
        tpr = (similarities.diag()>threshold).sum()/(similarities.diag().shape[0])
        fpr = (similarities[~np.eye(similarities.shape[0],dtype=bool)]>threshold).sum()/similarities[~np.eye(similarities.shape[0],dtype=bool)].shape[0]
        roc_points.append([fpr.item(), tpr.item()])
    roc_points.sort(key=lambda x:x[0])

    # get ranks
    big_matrix = np.stack([similarities.cpu(),np.eye(similarities.shape[0])],axis=-1)
    ranks = []
    for row_idx in (range(big_matrix.shape[0])):
        rank = int(np.where(np.array(sorted(big_matrix[row_idx].tolist(), key=lambda x: -x[0]))[:,1]==1)[0])
        ranks.append(rank)
    sorted_ranks = np.array(sorted(ranks))
    cum_frac = np.array([np.sum(sorted_ranks<=x)/len(sorted_ranks) for x in np.arange(len(sorted_ranks))])

    # record extra stats
    val_stats['fpr_at_one_percent_idx'] = [x[1] for x in roc_points if x[0] > 0.01][0]
    val_stats['equal_error_rate'] = [x[0] for x in roc_points if x[0] > (1-x[1])][0]
    val_stats['auc'] = auc([x[0] for x in roc_points],[x[1] for x in roc_points])
    print(label)
    plt.figure(roc_fig.number)
    roc_ax.plot([x[0] for x in roc_points],[x[1] for x in roc_points],label=label)

    plt.figure(rank_fig.number)
    rank_ax.plot(np.arange(1,len(sorted_ranks)+1),cum_frac, label=label)
    print(val_stats)

# make ROC plots
plt.figure(roc_fig.number)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(title)
plt.gca().set_aspect('equal')
plt.plot([0,1],[1,0], c='gray', linestyle='--', linewidth=1)
plt.plot([0,0.05,0.05],[0.95,0.95,1], c='black', linestyle='-', linewidth=1)
plt.legend(loc='lower right')
plt.tight_layout()

plt.xlim([0,0.05])
plt.ylim([0.95,1])
plt.title(title+', Zoomed')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(f'./figures/{save_name}_roc_curves_zoomed'+extension)

# make recall plots
plt.figure(rank_fig.number)
plt.plot([1,20,20],[0.85,0.85,1], c='black', linestyle='-', linewidth=1)
plt.tight_layout()
plt.ylim(0,1)
plt.xlim(1,2000)
plt.title(f"Retrieval Performance")
plt.xlabel('K')
plt.ylabel("Recall at K")
plt.legend(loc='lower right')
plt.gca().set_aspect(2000)
plt.savefig(f'./figures/{save_name}_rank_curves'+extension)
# zoomed version
plt.xlim([1,20])
plt.ylim([0.85,1])
plt.title('Retrieval Performance, Zoomed')
plt.gca().set_aspect(20/0.15)
plt.tight_layout()
plt.savefig(f'./figures/{save_name}_rank_curves_zoomed'+extension)
plt.close('all')
