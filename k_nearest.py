from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def generate_data(dataset):
    if dataset == "Circles":
        X, y = make_circles(n_samples=250, noise=0.2, random_state=54)
    elif dataset == "Moons":
        X, y = make_moons(n_samples=650, noise=0.3, random_state=18)
    elif dataset == "Blobs":
        X, y = make_blobs(n_samples=740, centers=2, random_state=44)
    elif dataset == "Anisotropic":
        X, y = make_classification(n_samples=810, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=11)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    elif dataset == "Varied":
        X, y = make_classification(n_samples=440, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=65)
        transformation = [[0.4, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:
        raise ValueError("Invalid dataset type")
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_holdout(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_random_subsampling(clf, X_train, y_train, X_test, y_test):
    accuracies = []
    for _ in range(10):  # 10 random splits
        X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, test_size=0.3)
        clf.fit(X_train_sub, y_train_sub)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    accuracy = np.mean(accuracies)
    return accuracy

def evaluate_k_fold_cross_validation(clf, X_train, y_train):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    accuracy = np.mean(scores)
    return accuracy

def evaluate_leave_one_out(clf, X_train, y_train):
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X_train, y_train, cv=loo)
    accuracy = np.mean(scores)
    return accuracy

def evaluate_bootstrap(clf, X_train, y_train, X_test, y_test):
    accuracies = []
    for _ in range(10):  # 10 bootstrap samples
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot = X_train[indices]
        y_train_boot = y_train[indices]
        clf.fit(X_train_boot, y_train_boot)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    accuracy = np.mean(accuracies)
    return accuracy

def main():
    datasets = ["Circles", "Moons", "Blobs", "Anisotropic", "Varied"]
    classifiers = {'k-Nearest Neighbors': KNeighborsClassifier()}
    normalization_methods = {'Min-max': MinMaxScaler(), 'Z-score': StandardScaler()}

    results = []
    for dataset in datasets:
        X, y = generate_data(dataset)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        for clf_name, clf in classifiers.items():
            for norm_name, norm in normalization_methods.items():
                X_train_normalized = norm.fit_transform(X_train)
                X_test_normalized = norm.transform(X_test)

                # Holdout
                accuracy_holdout = evaluate_holdout(clf, X_train_normalized, X_test_normalized, y_train, y_test)
                results.append((dataset, clf_name, norm_name, 'Holdout', accuracy_holdout))

                # Random subsampling
                accuracy_subsampling = evaluate_random_subsampling(clf, X_train_normalized, y_train, X_test_normalized, y_test)
                results.append((dataset, clf_name, norm_name, 'Random subsampling', accuracy_subsampling))

                # k-fold cross-validation
                accuracy_kfold = evaluate_k_fold_cross_validation(clf, X_train_normalized, y_train)
                results.append((dataset, clf_name, norm_name, 'k-fold cross-validation', accuracy_kfold))

                # Leave-one-out cross-validation
                accuracy_leave_one_out = evaluate_leave_one_out(clf, X_train_normalized, y_train)
                results.append((dataset, clf_name, norm_name, 'Leave-one-out', accuracy_leave_one_out))

                # Bootstrap
                accuracy_bootstrap = evaluate_bootstrap(clf, X_train_normalized, y_train, X_test_normalized, y_test)
                results.append((dataset, clf_name, norm_name, 'Bootstrap', accuracy_bootstrap))

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
