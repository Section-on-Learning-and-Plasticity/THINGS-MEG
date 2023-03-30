#
# Created on Thu Jan 26 2023 by Lina Teichmann
# Contact: lina.teichmann@nih.gov
#

import numpy as np
import mne, os
import pandas as pd 
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#*****************************#
### HELPER FUNCTIONS ###
#*****************************#

def load_epochs(bids_dir,participant,behav):
    epochs = mne.read_epochs(f'{bids_dir}/derivatives/preprocessed/preprocessed_P{participant}-epo.fif',preload=True)
    # THINGS-category number & image number start at 1 (matlab based) so subtracting 1 to use for indexing
    epochs.metadata['things_image_nr'] = epochs.metadata['things_image_nr']-1
    epochs.metadata['things_category_nr'] = epochs.metadata['things_category_nr']-1

    # adding dimensional weights to metadata
    epochs.metadata[['dim'+ str(d+1) for d in range(66)]] = np.nan
    for i in range(len(epochs.metadata)):
        if not np.isnan(epochs.metadata.loc[i,'things_image_nr']):
            epochs.metadata.loc[i,['dim'+ str(d+1) for d in range(66)]] = behav[int(epochs.metadata.loc[i,'things_image_nr']),:]

    return epochs

def train_test_split(epo_stacked,test_part):
    epochs_test=epo_stacked[f'participant == {test_part} and trial_type == "exp"']
    dim_columns = [col for col in epochs_test.metadata if col.startswith('dim')]
    y_test = epochs_test.metadata[dim_columns].to_numpy()
    x_test = epochs_test._data

    epochs_train=epo_stacked[f'participant != {test_part} and trial_type == "exp"']
    dim_columns = [col for col in epochs_train.metadata if col.startswith('dim')]
    y_train = epochs_train.metadata[dim_columns].to_numpy()
    x_train = epochs_train._data

    return x_train, x_test, y_train, y_test

def run_linear_regression(x_train,y_train,x_test,y_test,n_ys):
    pipe = Pipeline([('scaler', StandardScaler()), 
            ('regression', LinearRegression())])
    
    pipe.fit(x_train,y_train)
    predictions = pipe.predict(x_test)

    return [np.corrcoef(predictions[:,d],y_test[:,d])[0,1] for d in range(n_ys)]


#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bids_dir",
        required=True,
        help='path to bids root',
    )

    args = parser.parse_args()

    # folders
    bids_dir = args.bids_dir

    behav = np.loadtxt(f'{bids_dir}/sourcedata/meg_paper/predictions_66d_elastic_clip-ViT-B-32_visual_THINGS.txt')
    res_dir = f'{bids_dir}/derivatives/meg_paper/output/regression/'
    perm_dir = f'{bids_dir}/derivatives/meg_paper/output/regression/permutations/'

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(perm_dir):
        os.makedirs(perm_dir)


    # parameters
    n_sessions = 12
    n_features = 271
    n_ys = 66
    n_exp_trials = 1854
    n_times = 281
    n_participants = 4
    
    # load data
    for p in range(1,n_participants+1):
        print('loading data participant ' + str(p))
        epochs_1 = load_epochs(bids_dir,p,behav)
        epochs_1.metadata['participant'] = p

        if p!=1:
            epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
            epochs = mne.concatenate_epochs([epochs,epochs_1])
        else:
            epochs = epochs_1.copy()

    
    # train-test splits for all cross-validations
    x_train = np.zeros((n_exp_trials*n_sessions*(n_participants-1),n_features,n_times,n_participants))
    x_test = np.zeros((n_exp_trials*n_sessions,n_features,n_times,n_participants))
    y_train = np.zeros((n_exp_trials*n_sessions*(n_participants-1),n_ys,n_participants))
    y_test = np.zeros((n_exp_trials*n_sessions,n_ys,n_participants))

    for i,test_part in enumerate(range(1,n_participants+1)):
        x_train[:,:,:,i], x_test[:,:,:,i], y_train[:,:,i], y_test[:,:,i] = train_test_split(epochs,test_part)
    print('made train and test sets for all cv-folds ')

    # run regression model for each cross-validation split
    corr_dims_cv = np.zeros((n_times,n_ys,n_participants))
    for i,test_part in enumerate(range(1,n_participants+1)):
        print('run regression cv: ' + str(test_part))
        x_train, x_test, y_train, y_test = train_test_split(epochs,test_part)
        res = Parallel(n_jobs=48, prefer="threads")(delayed(run_linear_regression)(x_train[:,:,t],y_train,x_test[:,:,t],y_test,n_ys) for t in range(len(epochs.times)))
        corr_dims_cv[:,:,i] = np.array(res)
    corr_dims_ts = np.mean(corr_dims_cv,axis=2)
    pd.DataFrame(corr_dims_ts).to_csv(f'{res_dir}/Linreg_cross_predict-dims.csv')

