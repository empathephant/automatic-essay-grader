import os

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import features

def get_data():
    filename = os.path.join('./data', "feature_data.tsv")

    if os.path.isfile(filename):
        print("Features already analysed.")
    else:
        print("Analysing features...")
        features.get_feature_data()
        print("Analysis complete.")

def load_data():
    get_data()
    # Load dataset
    url = "./data/feature_data.tsv"
    names = ['FEAT1', 'FEAT2', 'FEAT3', 'FEAT4', 'FEAT5', 'FEAT6', 'FEAT7', 'FEAT8', 'FEAT9', 'FEAT10',
             'FEAT11', 'FEAT12', 'FEAT13', 'FEAT14', 'FEAT15', 'FEAT16', 'FEAT17', 'FEAT18', 'FEAT19', 'FEAT20',
             'FEAT21', 'SCORE']
    dataset = pandas.read_csv(url, sep='\t', names=names, skiprows=[0])
    return dataset

def print_data_info(dataset):
    descrip_filename = os.path.join('./data', "data_description.txt")

    if os.path.isfile(descrip_filename):
        print("Data description already written.")
    else:
        print(f'Writing data description to {descrip_filename}...')

        with open(descrip_filename, "w+") as f:
            f.write("##############################################\n")

            # shape
            f.write("Shape of dataset:\n\n")
            f.write(repr(dataset.shape))
            f.write("\n##############################################\n")

            # head
            f.write("Head of dataset:\n\n")
            f.write(repr(dataset.head(20)))
            f.write("\n##############################################\n")

            #descriptions
            f.write("Description of dataset:\n\n")
            f.write(repr(dataset.describe()))
            f.write("\n##############################################\n")

            # class distribution
            f.write("Score distribution for dataset:\n\n")
            f.write(repr(dataset.groupby('SCORE').size()))
            f.write("\n##############################################\n")

        print("Description complete.")

def create_visualizations(dataset):
    filename = './data/plots'

    if os.path.isdir(filename):
        print("Plots already drawn.")
    else:
        print("Drawing plots...")
        os.mkdir("./data/plots")

        # univariate plots
        # box and whisker plots
        dataset.plot(kind='box', subplots=True, layout=(5, 5), sharex=False, sharey=False)
        plt.savefig(os.path.join('./data/plots', "box_and_whisker.png"))

        # histograms
        dataset.hist()
        plt.savefig(os.path.join('./data/plots', "histogram.png"))

        # multivariate plots
        # scatter plot matrix
        scatter_matrix(dataset)
        plt.savefig(os.path.join('./data/plots', "scatter_plot.png"))

        print("Plot creation complete.")

def create_validation_set(dataset):
    print("Creating validation set...")
    #Split out validation dataset
    value_array = dataset.values
    X = value_array[:, 0:21]
    Y = value_array[:, 21]
    validation_size = 0.30
    seed = 42
    datasets = {}
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    datasets.update({'X_tr': X_train, 'X_val': X_validation, 'Y_tr': Y_train, 'Y_val': Y_validation})
    print("Validation set created.")
    return datasets

def evaluate_algorithms(datasets):
    filename = os.path.join('./data', "algorithm_eval.txt")

    if os.path.isfile(filename):
        print("Algorithms already evaluated.")
    else:
        print("Evaluating algorithms...")

        # Test options and evaluation metric
        seed = 42
        scoring = 'accuracy'

        #Spot check algoritms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        #Evaluate each model in turn
        results = []
        names = []

        with open(filename, "w+") as f:
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, datasets['X_tr'], datasets['Y_tr'], cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                report = f'{name}: {cv_results.mean()} ({cv_results.std()})\n'
                f.write(report)

        # Compare algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig(os.path.join('./data/plots', "compare_algorithms.png"))

def build_machine_learning():
    dataset = load_data()

    print_data_info(dataset)
    create_visualizations(dataset)
    datasets = create_validation_set(dataset)
    evaluate_algorithms(datasets)

