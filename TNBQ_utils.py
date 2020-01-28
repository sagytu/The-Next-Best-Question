import numpy as np
from scipy.spatial import distance
from scipy.linalg.blas import dgemm
from sklearn.preprocessing import MinMaxScaler
import shap
from tqdm import tqdm

def K_nearest_neighbors(test_data, x_train, k, common, metric='euclidean', return_inverse_dists=False):

    to_search = test_data.values
    x_train_common_features = x_train.values[:, common]

    # Two ways to compute distance -
    # SciPy euclidean distance (faster than NumPy's)
    dists = distance.cdist(to_search.reshape(1, to_search.shape[0]), x_train_common_features, metric=metric)

    # SciPy BLAS euclidean distance (works faster for sparse matrices (but not for random))
    # ts = to_search.reshape(1, to_search.shape[0])
    # dists = dgemm(alpha=-2, a=ts, b=x_train_common_features,trans_b=True)
    # dists += np.einsum('ij,ij->i', ts,ts)[:, None]
    # dists += np.einsum('ij,ij->i', x_train_common_features, x_train_common_features)
    # dists = np.sqrt(dists)

    # Taking the K closest neighbors
    indices = np.argpartition(dists, kth=k)[:, :k]

    if return_inverse_dists:
        dists = dists[0][indices[0]]
        weights = 1 / dists

        # Compromise decision - if the distance is 0 (exact same values) set the inverse-distance weight to the maximum
        # weight (closest neighbor).
        weights[np.isinf(weights)] = -1
        weights[weights == -1] = np.max(weights)

        # If all distances are 0 (now changed to -1), set all weights to 1
        if np.sum(weights) == -k:
            weights = np.ones(k)

        return indices, weights

    return indices


def radius_neighborhood(test_data, x_train, common, metric='euclidean', dist_th=0.5, min_neighborhood_size=15, max_neigborhood_size=100):


    to_search = test_data.values
    x_train_common_features = x_train.values[:, common]

    # Using SciPy euclidean distance (faster than NumPy's)
    dists = distance.cdist(to_search.reshape(1, to_search.shape[0]), x_train_common_features, metric=metric)

    max_neigborhood_size = min(max_neigborhood_size, x_train.shape[0]-1)
    # Taking the maximum neighborhood size - they will be further examined for distance
    indices = np.argpartition(dists, kth=max_neigborhood_size)[:, :max_neigborhood_size]

    # Keep only those under dist_th
    x_train_max_neighborhood = x_train.values[indices[0], :][:, common]
    dists = distance.cdist(to_search.reshape(1, to_search.shape[0]), x_train_max_neighborhood, metric=metric)

    # As features acquired, the distance span changes, so squishing to 0,1 range
    sc = MinMaxScaler()
    dists = sc.fit_transform(dists[0].reshape(-1,1))
    dists = dists.reshape(dists.shape[1], dists.shape[0])

    indices_within_th = indices[0][dists[0] < dist_th]

    # check that the neighborhood contain at least min_neighborhood_size neighbors
    if len(indices_within_th) < min_neighborhood_size:
        idx_dist_dict = dict(zip(indices[0], dists[0]))
        indices = sorted(idx_dist_dict.keys(), key=lambda x: idx_dist_dict[x])
        return indices[:min_neighborhood_size]

    return indices_within_th


already_calculated_shap_idxs = {}


def feature_importance_by_KNN(X_train, y_test, model, knn_idxs, masked_data_filling, verbose=False, knn_weights=None,
                              scale_range=(0.2,1)):

    is_multiclass = False
    assert len(y_test.value_counts()) >= 2

    if len(y_test.value_counts()) > 2 and model._estimator_type != 'regressor':
        is_multiclass = True

    # Full list of features and explainer
    feat_names = X_train.columns
    explainer = shap.TreeExplainer(model)

    res = {}
    global already_calculated_shap_idxs

    # for each row in the test set (=the length of knn_idxs)
    for i in (tqdm(range(len(knn_idxs))) if verbose else range(len(knn_idxs))):

        # Create a dataframe from the k nearest neighbors of a row (index) i, and compute its SHAP values
        neighbors_idxs_to_calc = knn_idxs[i]

        # Filter the samples that their SHAP values were already calculation (for runtime reasons)
        neighbors_idxs_to_calc = [j for j in neighbors_idxs_to_calc if j not in already_calculated_shap_idxs.keys()]

        # Add weights if available
        if knn_weights is not None:
            transformer = MinMaxScaler(feature_range=scale_range)
            norm_weights = transformer.fit_transform(knn_weights[i].reshape(-1, 1))
            norm_weights = norm_weights.reshape(norm_weights.shape[0], )
        else:
            # If no weights given, set equal weights (just ones)
            norm_weights = np.ones(len(knn_idxs[0]))

        if is_multiclass:
            # If has samples with no SHAP values yet, construct a DataFrame and calculate them
            if len(neighbors_idxs_to_calc) > 0:
                sliced_df = X_train.iloc[neighbors_idxs_to_calc]
                shap_values = np.array(explainer.shap_values(sliced_df))
                for j, neighbor_i in enumerate(neighbors_idxs_to_calc):
                    # Multilabel edit - for each label ([:,_]), take the SHAP values of sample j ([_,j])
                    already_calculated_shap_idxs[neighbor_i] = shap_values[:, j]

            # Gather again the SHAP values of the neighbors, sort the features names (by these values) and keep only the
            # masked features.
            # Multilabel edit - convert to the same shape of the original SHAP values matrix - (#labels, #samples, #features)
            shap_values = []
            for lbl in range(model.n_classes_):
                shap_values.append([])

            for w_i, s_v_i in zip(norm_weights, knn_idxs[i]):
                for lbl in range(model.n_classes_):
                    # Multiplying SHAP values of a neighbor (s_v_i) by its scaled IDW (w_i)
                    # (An experiment showed that multiplying by the IDW as-is does not improve the results)
                    shap_values[lbl].append(already_calculated_shap_idxs[s_v_i][lbl] * w_i)
            # shap_values = [already_calculated_shap_idxs[s_v_i] for s_v_i in knn_idxs[i]]
            shap_values = np.array(shap_values)
            shap_values = np.sum(np.average(np.abs(shap_values), axis=1), axis=0)

        else:
            # If has samples with no SHAP values yet, construct a DataFrame and calculate them
            if len(neighbors_idxs_to_calc) > 0:
                sliced_df = X_train.iloc[neighbors_idxs_to_calc]
                shap_values = explainer.shap_values(sliced_df)
                for j, neighbor_i in enumerate(neighbors_idxs_to_calc):
                    already_calculated_shap_idxs[neighbor_i] = shap_values[j]

            # Gather again the SHAP values of the neighbors, sort the features names (by these values) and keep only the
            # masked features.
            shap_values = [already_calculated_shap_idxs[s_v_i] * w_i for s_v_i, w_i in zip(knn_idxs[i],norm_weights)]
            shap_values = np.average(np.abs(shap_values), axis=0)


        feats_shaps_dict = dict(zip(feat_names, shap_values))
        sorted_feats = sorted(feats_shaps_dict.keys(), key=lambda f: feats_shaps_dict[f], reverse=True)
        sorted_feats = [f for f in sorted_feats if f in masked_data_filling[i].keys()]

        res[i] = sorted_feats

    return res
