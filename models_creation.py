from xgboost import XGBClassifier


def train_model(dataset, X_train, y_train, _seed):
    if dataset.lower() == 'mnist':
        return train_model_mnist(X_train, y_train, _seed)
    elif dataset.lower() == 'miniboone':
        return train_model_miniboone(X_train, y_train, _seed)
    elif dataset.lower() == 'parkinson':
        return train_model_parkinson(X_train, y_train, _seed)
    elif dataset.lower() == 'cardio':
        return train_model_cardio(X_train, y_train, _seed)
    elif dataset.lower() == 'statlog':
        return train_model_statlog(X_train, y_train, _seed)
    elif dataset.lower() == 'spambase':
        return train_model_spambase(X_train, y_train, _seed)


def train_model_cardio(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=75, max_depth=5, random_state=_seed)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_model_parkinson(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=50, max_depth=15, gamma=0.05)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_model_mnist(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=15, max_depth=10, random_state=_seed)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_model_miniboone(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=25, max_depth=15, random_state=_seed)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_model_statlog(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=50, max_depth=10)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_model_spambase(X_train, y_train, _seed):
    xgb_clf = XGBClassifier(n_estimators=75, max_depth=5, random_state=_seed)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf
