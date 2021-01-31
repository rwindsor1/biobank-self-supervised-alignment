import sys, os, glob
import pandas as pd

df = pd.read_csv('biobank_mri_dxa_id_alignments.csv')

scan_dfs  = {'train': pd.read_csv('biobank_mri_dxa_id_train.csv'), 
             'val'  : pd.read_csv('biobank_mri_dxa_id_val.csv'),
             'test' : pd.read_csv('biobank_mri_dxa_id_test.csv')} 

for key in scan_dfs:
    aligned_df = pd.merge(df,scan_dfs[key])
    aligned_df.drop(list(aligned_df.columns)[:3],axis=1, inplace=True)
    aligned_df.to_csv('biobank_mri_dxa_id_alignments_' + key + '.csv')

