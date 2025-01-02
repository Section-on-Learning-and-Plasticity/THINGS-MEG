#
# Created on Thu Jan 26 2023 by Lina Teichmann
# Modified on Wed Oct 2 2024 by Lina Teichmann
# Contact: lina.teichmann@nih.gov
#

import numpy as np
import mne, os
import pandas as pd 
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV


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

def train_test_split(epo_stacked,test_sess):
    epochs_test=epo_stacked[f'session_nr == {test_sess} and trial_type == "exp"']
    dim_columns = [col for col in epochs_test.metadata if col.startswith('dim')]
    y_test = epochs_test.metadata[dim_columns].to_numpy()
    x_test = epochs_test._data

    epochs_train=epo_stacked[f'session_nr != {test_sess} and trial_type == "exp"']
    dim_columns = [col for col in epochs_train.metadata if col.startswith('dim')]
    y_train = epochs_train.metadata[dim_columns].to_numpy()
    x_train = epochs_train._data

    return x_train, x_test, y_train, y_test

def linear_regression(x_train,y_train,x_test):
    pipe = Pipeline([('scaler', StandardScaler()), 
            ('regression', RidgeCV(alphas=np.linspace(1e-5, 30000, 100),
                                   alpha_per_target=True))])
    
    pipe.fit(x_train,y_train)
    predictions = pipe.predict(x_test)

    return predictions

def run_regression_dims(bids_dir,participant,test_sess,n_sessions = 12,n_features = 271,n_ys = 66,n_exp_trials = 1854,n_times = 281):
    # load data
    print('loading data participant ' + str(participant))
    epochs = load_epochs(bids_dir,participant,behav)
    
    # divide data in train-test split for this cross-validation
    x_train, x_test, y_train, y_test = train_test_split(epochs,test_sess)
    print(f'made train and test set: leaving out session {test_sess} as testing')

    # run regression model 
    corr_dims_cv = np.zeros((n_times,n_ys))
    y_pred = Parallel(n_jobs=48, prefer="threads")(delayed(linear_regression)(x_train[:,:,t],y_train,x_test[:,:,t]) for t in range(len(epochs.times)))
    y_pred, weights, scaled_x_train, scaled_x_test = zip(*res)

    np.save(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_ypred_cv{test_sess}.npy', np.array(y_pred))
    np.save(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_ytrue_cv{test_sess}.npy', np.array(y_test))
   
    # run correlation
    res = []
    for  t in range(np.array(y_pred).shape[0]):
        print(t)
        res.append([np.corrcoef(np.array(y_pred)[t,:,d],y_test[:,d])[0,1] for d in range(n_ys)])
    pd.DataFrame(np.array(res)).to_csv(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_cv{test_sess}.csv')

    print(f'DONE! cross-validation split {test_sess} out of {n_sessions}')


#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-participant",
        required=True,
        help='participant bids ID (e.g., 1)',
    )

    parser.add_argument(
        "-bids_dir",
        required=True,
        help='path to bids root',
    )

    parser.add_argument(
        "-test_chunk",
        required=False,
        help='specify session number that should be left out for testing in the cross-validation',
        type=int,
        default = 0,
    )


    args = parser.parse_args()

    # folders
    bids_dir = args.bids_dir
    participant = args.participant
    test_sess = args.test_chunk

    behav = np.loadtxt(f'{bids_dir}/sourcedata/meg_paper/predictions_66d_elastic_clip-ViT-B-32_visual_THINGS.txt')
    res_dir = f'{bids_dir}/derivatives/meg_paper/output/regression/'
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    run_regression_dims(bids_dir,participant,test_sess)

