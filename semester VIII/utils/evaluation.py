import numpy as np

def compute_metrics(users_idx: np.ndarray, test_interactions: dict, recs: np.ndarray):
    k = recs.shape[1]
    precision = recall = map = 0.
    
    for i, user_idx in enumerate(users_idx):
        user_inters = test_interactions[user_idx]
        user_recs = recs[i]
        relevant_count = len(user_inters.intersection(user_recs))

        precision += relevant_count / k
        recall += relevant_count / len(user_inters)

        is_relevant = np.in1d(user_recs, list(user_inters)).astype(int)
        rels_idx = is_relevant.nonzero()
        is_relevant[rels_idx] = is_relevant[rels_idx].cumsum()
        denum = np.arange(1, k+1)
        mapi = (is_relevant / denum).sum() / k
        
        map += mapi

    precision /= len(users_idx)
    recall /= len(users_idx)
    map /= len(users_idx)

    return precision, recall, map