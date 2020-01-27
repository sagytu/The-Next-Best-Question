import numpy as np
import copy
import shap
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from TNBQ_utils import K_nearest_neighbors, feature_importance_by_KNN, radius_neighborhood


def global_shap_baseline(X_train, X_test_masked, y_test, model, n_masked, mask_data_filling, metric='accuracy'):
    """
    Adds features according to their global SHAP values (= over all training data).
    This is the traditional feature importance using SHAP values and used as baseline for the experiments.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    X_test_masked: pd.DataFrame
        Masked test set.
    y_test : pd.DataFrame
        True labels of the test set.
    model : xgboost.XGBClassifier or sklearn.ensemble.RandomForestClassifier
        Classification model.
    n_masked : int
        Number of masked features.
    mask_data_filling : dictionary
        The filling of the masked data. (the product of mask_data)
    shap_precalculation : str
        A path to a pre-calculated Shap matrix (as .npy file)
    Returns
    -------
    list :
        A list of calculated metric (AUC or accuracy for binary of multi-class respectively) after every iteration.
    """

    if metric.lower() == 'auc':
        metric_func = roc_auc_score
    else:
        metric_func = accuracy_score
    assert len(y_test.value_counts()) >= 2

    X_test_masked = X_test_masked.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if len(y_test.value_counts()) > 2:
        is_multiclass = True
    else:
        is_multiclass = False

    if is_multiclass:
        f_dict = dict(zip(X_train.columns, np.sum(np.average(np.abs(shap_values), axis=1), axis=0)))
    else:
        f_dict = dict(zip(X_train.columns, np.average(np.abs(shap_values), axis=0)))

    # feature list sorted by the SHAP values over the whole data (train set)
    global_features_sorted = sorted(f_dict.keys(), key=lambda x: f_dict[x], reverse=True)
    feats_to_add = {}

    # Create for each sample its own list (= his masked features sorted by the global SHAP calculation)
    for r_i in range(X_test_masked.shape[0]):
        f_to_add_lst = [f for f in global_features_sorted if f in mask_data_filling[r_i].keys()]
        feats_to_add[r_i] = f_to_add_lst
    acc_lst = []

    # Check AUC with all masked feature.
    acc_lst.append(metric_func(y_test, model.predict(X_test_masked)))

    # Iterate over features and samples. For each sample, in each iteration - add one feature.
    # After adding to all samples, calculate AUC and move on to the next masked feature.
    for feat_i in range(n_masked):
        for rec_i in range(X_test_masked.shape[0]):

            # If the sample still has masked features - add one
            if feat_i < len(feats_to_add[rec_i]):
                f_name = feats_to_add[rec_i][feat_i]
                X_test_masked.set_value(rec_i, f_name, mask_data_filling[rec_i][f_name])

        acc_lst.append(metric_func(y_test, model.predict(X_test_masked)))
        print(acc_lst)
    return acc_lst


def the_next_best_question(X_train, X_test_masked, y_test, model, n_masked, mask_data_filling, K, metric='accuracy'):
    """
    Adds features using the "Next Best Question" - SHAP evaluation over K nearest neighbors.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    X_test_masked: pd.DataFrame
        Masked test set.
    y_test : pd.DataFrame
        True labels of the test set.
    model : xgboost.XGBClassifier or sklearn.ensemble.RandomForestClassifier
        Classification model.
    n_masked : int
        Number of masked features.
    mask_data_filling : dictionary
        The filling of the masked data. (the product of mask_data)
    K : int
        Number of neighbors (from X_train) to use for SHAP calculation.
    stop : int
        Number of features to add (stops the loop afterwards).
    Returns
    -------
    list :
        A list of calculated metric (AUC or accuracy for binary of multi-class respectively) after every iteration.
    """
    if metric.lower() == 'auc':
        metric_func = roc_auc_score
    else:
        metric_func = accuracy_score
    X_train, X_test_masked, y_test = X_train.copy(), X_test_masked.copy(), y_test.copy()
    mask_data_filling = copy.deepcopy(mask_data_filling)

    assert len(y_test.value_counts()) >= 2


    print('Adding Iterative-KNN using SHAP')
    # calculate initial auc with n_masked features masked
    acc_knn = []
    acc_knn.append(metric_func(y_test, model.predict(X_test_masked)))

    for _i in range(n_masked):
        print('Adding feature number: ' + str(_i+1))

        knn_idxs = None
        for rec_i in tqdm(range(X_test_masked.shape[0])):
            available_features = [f for f in X_train.columns if f not in mask_data_filling[rec_i]]
            row = X_test_masked.iloc[rec_i][available_features]
            if knn_idxs is None:
                knn_idxs = K_nearest_neighbors(row, X_train, K, common=available_features)
            else:
                knn_idxs = np.append(knn_idxs, K_nearest_neighbors(row, X_train, K, common=available_features),
                                 axis=0)

        feats_to_add = feature_importance_by_KNN(X_train, y_test, model, knn_idxs, mask_data_filling, verbose=True)

        # Adding only 1 feature for each sample and removing it from its masked features dictionary.
        # Then repeating the whole process (KNN, feature importance, adding...)
        for rec_i in range(X_test_masked.shape[0]):

            masked_features_by_importance = feats_to_add[rec_i]
            if len(mask_data_filling) > 0:

                # Adding the most important feature
                most_valuable_f = masked_features_by_importance[0]
                X_test_masked.set_value(rec_i, most_valuable_f, mask_data_filling[rec_i][most_valuable_f])

                # after adding this feature, it is not masked anymore
                del mask_data_filling[rec_i][most_valuable_f]

        acc_knn.append(metric_func(y_test, model.predict(X_test_masked)))

        if _i % 25 == 0:
            print('Feature', _i, 'results:')
            print(acc_knn)

        # NEED TO REMOVE THIS - JUST A LIMITATION FOR HIGH DIMENSIONAL DATASETS --->
        if _i == 30:
            break
    return acc_knn


def the_next_best_question_weighted(X_train, X_test_masked, y_test, model, n_masked, mask_data_filling, K, scale_range=(0.2,1), metric='accuracy'):
    """
    Adds features using the "Next Best Question" - SHAP evaluation over K nearest neighbors.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    X_test_masked: pd.DataFrame
        Masked test set.
    y_test : pd.DataFrame
        True labels of the test set.
    model : xgboost.XGBClassifier or sklearn.ensemble.RandomForestClassifier
        Classification model.
    n_masked : int
        Number of masked features.
    mask_data_filling : dictionary
        The filling of the masked data. (the product of mask_data)
    K : int
        Number of neighbors (from X_train) to use for SHAP calculation.
    stop : int
        Number of features to add (stops the loop afterwards).
    scale_range : tuple
        Tuple of the MinMax scaling bounds.
    Returns
    -------
    list :
        A list of calculated metric (AUC or accuracy for binary of multi-class respectively) after every iteration.
    """
    if metric.lower() == 'auc':
        metric_func = roc_auc_score
    else:
        metric_func = accuracy_score
    X_train, X_test_masked, y_test = X_train.copy(), X_test_masked.copy(), y_test.copy()
    mask_data_filling = copy.deepcopy(mask_data_filling)

    assert len(y_test.value_counts()) >= 2

    print('Adding Iterative-KNN using SHAP')
    # calculate initial auc with n_masked features masked
    acc_lst = []
    acc_lst.append(metric_func(y_test, model.predict(X_test_masked)))

    for _i in range(n_masked):
        print('Adding feature number: ' + str(_i+1))

        knn_idxs = None
        knn_weights = None
        for rec_i in tqdm(range(X_test_masked.shape[0])):
            available_features = [f for f in X_train.columns if f not in mask_data_filling[rec_i]]
            row = X_test_masked.iloc[rec_i][available_features]
            if knn_idxs is None:
                knn_idxs, knn_weights = K_nearest_neighbors(row, X_train, K, common=available_features, return_inverse_dists=True)
                knn_weights = np.array([knn_weights])
            else:
                indices, weights = K_nearest_neighbors(row, X_train, K, common=available_features, return_inverse_dists=True)
                weights = np.array([weights])
                knn_idxs = np.append(knn_idxs, indices, axis=0)
                knn_weights = np.append(knn_weights, weights, axis=0)

        feats_to_add = feature_importance_by_KNN(X_train, y_test, model, knn_idxs, mask_data_filling, verbose=True,
                                                 knn_weights=knn_weights, scale_range=scale_range)

        # Adding only 1 feature for each sample and removing it from its masked features dictionary.
        # Then repeating the whole process (KNN, feature importance, adding...)
        for rec_i in range(X_test_masked.shape[0]):

            masked_features_by_importance = feats_to_add[rec_i]
            if len(mask_data_filling) > 0:

                # Adding the most important feature
                most_valuable_f = masked_features_by_importance[0]
                X_test_masked.set_value(rec_i, most_valuable_f, mask_data_filling[rec_i][most_valuable_f])

                # after adding this feature, it is not masked anymore
                del mask_data_filling[rec_i][most_valuable_f]

        acc_lst.append(metric_func(y_test, model.predict(X_test_masked)))
        print(acc_lst)
    return acc_lst


def the_next_best_question_radius_neighborhood(X_train, X_test_masked, y_test, model, n_masked, mask_data_filling,
                                               dist_th=0.5, min_neigh_size=15, max_neigh_size=100, metric='accuracy'):
    """
    Adds features using the "Next Best Question" - SHAP evaluation over K nearest neighbors.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    X_test_masked: pd.DataFrame
        Masked test set.
    y_test : pd.DataFrame
        True labels of the test set.
    model : xgboost.XGBClassifier or sklearn.ensemble.RandomForestClassifier
        Classification model.
    n_masked : int
        Number of masked features.
    mask_data_filling : dictionary
        The filling of the masked data. (the product of mask_data)
    K : int
        Number of neighbors (from X_train) to use for SHAP calculation.
    stop : int
        Number of features to add (stops the loop afterwards).
    dist_th : float
        The distance threshold to the neighborhood radius.
    min_neigh_size : int
        The minimum number of samples in the neighborhood to pick.
    max_neigh_size : int
        The maximum number of samples in the neighborhood to keep.
    Returns
    -------
    list :
        A list of calculated metric (AUC or accuracy for binary of multi-class respectively) after every iteration.
    """
    if metric.lower() == 'auc':
        metric_func = roc_auc_score
    else:
        metric_func = accuracy_score
    X_train, X_test_masked, y_test = X_train.copy(), X_test_masked.copy(), y_test.copy()
    mask_data_filling = copy.deepcopy(mask_data_filling)

    assert len(y_test.value_counts()) >= 2

    print('Adding Iterative-KNN using SHAP')
    # calculate initial auc with n_masked features masked
    aucs_knn = []
    aucs_knn.append(metric_func(y_test, model.predict(X_test_masked)))

    for _i in range(n_masked):
        print('Adding feature number: ' + str(_i+1))

        # Neighborhoods might be in different length, so can't use an array
        knn_idxs = []
        for rec_i in tqdm(range(X_test_masked.shape[0])):
            available_features = [f for f in X_train.columns if f not in mask_data_filling[rec_i]]
            row = X_test_masked.iloc[rec_i][available_features]
            knn_idxs.append(radius_neighborhood(row, X_train, common=available_features, dist_th=dist_th,
                                                min_neighborhood_size=min_neigh_size, max_neigborhood_size=max_neigh_size))


        feats_to_add = feature_importance_by_KNN(X_train, y_test, model, knn_idxs, mask_data_filling, verbose=True)

        # Adding only 1 feature for each sample and removing it from its masked features dictionary.
        # Then repeating the whole process (KNN, feature importance, adding...)
        for rec_i in range(X_test_masked.shape[0]):

            masked_features_by_importance = feats_to_add[rec_i]
            if len(mask_data_filling) > 0:

                # Adding the most important feature
                most_valuable_f = masked_features_by_importance[0]
                X_test_masked.set_value(rec_i, most_valuable_f, mask_data_filling[rec_i][most_valuable_f])

                # after adding this feature, it is not masked anymore
                del mask_data_filling[rec_i][most_valuable_f]

        aucs_knn.append(metric_func(y_test, model.predict(X_test_masked)))
        print(aucs_knn)
    return aucs_knn


