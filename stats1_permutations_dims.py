import numpy as np 
def run_permutations(res_dir,participant,n_sessions=12):
    # make consistent permutation indices
    np.random.seed(1423)
    n_perms = 1000
    
    matrix = np.tile(np.arange(1854), (n_perms, 1))
    perm_idx = np.apply_along_axis(np.random.permutation, 1, matrix)

    # loop over all predicted and true y-values, permute y-values & re-run correlation
    corr_ridge_perm_all = []
    for cv in range(1,n_sessions+1):
        print(cv)
        y_pred = np.load(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_ypred_cv{cv}.npy')
        y_true = np.load(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_ytrue_cv{cv}.npy')

        corr_ridge_perm = []
        for perm in range(n_perms):
            print(f'perm {perm}')
            y_perm = y_true.copy()
            y_perm = y_perm[perm_idx[perm,:]]

            corr_t = []
            for t in range(y_pred.shape[0]):
                print(f'time {t}')
                corr_t.append([np.corrcoef(y_pred[t,:,d],y_perm[:,d])[0,1] for d in range(66)])
            corr_ridge_perm.append(np.array(corr_t))
            
        corr_ridge_perm_all.append(corr_ridge_perm)
        
    # save output    
    np.save(f'{res_dir}/P{participant}_ridge-reg_within_predict-dims_permutations.npy', np.array(corr_ridge_perm_all) )

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
    
    parser.add_argument(
        "-participant",
        required=True,
        help='participant number',
        type=int,
        default = 0,
    )

    args = parser.parse_args()

    # folders
    bids_dir = args.bids_dir
    participant = args.participant
     
    res_dir = f'{bids_dir}/derivatives/meg_paper/output/regression/'

    run_permutations(res_dir,participant)
