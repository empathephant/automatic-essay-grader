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
             'FEAT11', 'FEAT12', 'FEAT13', 'FEAT14', 'FEAT15', 'FEAT16', 'FEAT17', 'FEAT18',
             #  'FEAT19', 'FEAT20', 'FEAT21',
             'SCORE']
    dataset = pandas.read_csv(url, sep='\t', names=names, skiprows=[0])
    return dataset

def print_data_info(dataset):
    descrip_filename = os.path.join('./data', "data_description.txt")

    if os.path.isfile(descrip_filename):
        print("Data description already written.")
    else:
        print(f'Writing data description to {descrip_filename}...')

        with open(descrip_filename, "w+") as f:
            line_break = "\n##############################################\n"
            f.write(line_break)

            # shape
            f.write("Shape of dataset:\n\n")
            f.write(repr(dataset.shape))
            f.write(line_break)

            # head
            f.write("Head of dataset:\n\n")
            f.write(repr(dataset.head(20)))
            f.write(line_break)

            # descriptions
            f.write("Description of dataset:\n\n")
            f.write(repr(dataset.describe()))
            f.write(line_break)

            # class distribution
            f.write("Score distribution for dataset:\n\n")
            f.write(repr(dataset.groupby('SCORE').size()))
            f.write(line_break)

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
    # Split out validation dataset
    value_array = dataset.values
    X = value_array[:, 0:18]
    Y = value_array[:, 18]
    validation_size = 0.20
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

        # Spot check algoritms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))

        # Evaluate each model in turn
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

def check_validation_set(datasets):
    filename = os.path.join('./data', "accuracy_scores.txt")

    if os.path.isfile(filename):
        print("Accuracy already evaluated.")
    else:
        print("Evaluating accuracy of LR algorithm...")
        # Make predictions on validation dataset
        lr = LogisticRegression()
        lr.fit(datasets['X_tr'], datasets['Y_tr'])
        predictions = lr.predict(datasets['X_val'])

        with open(filename, "w+") as f:
            line_break = "\n##############################################\n"
            f.write(line_break)

            f.write('Accuracy score:\n')
            f.write(repr(accuracy_score(datasets['Y_val'], predictions)))
            f.write(line_break)

            f.write('Confusion matrix:\n')
            f.write(repr(confusion_matrix(datasets['Y_val'], predictions)))
            f.write(line_break)

            f.write('Classification report:\n')
            f.write(repr(classification_report(datasets['Y_val'], predictions)))
            f.write(line_break)

def build_machine_learning():
    dataset = load_data()

    print_data_info(dataset)
    create_visualizations(dataset)
    datasets = create_validation_set(dataset)
    evaluate_algorithms(datasets)

    check_validation_set(datasets)

