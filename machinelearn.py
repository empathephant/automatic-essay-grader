import os

import pandas
from pandas.tools.plotting import scatter_matrix
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
        print("Analysis complete")

def load_data():
    # Load dataset
    url = "./data/feature_data.tsv"
    names = ['FEAT1', 'FEAT2', 'FEAT3', 'FEAT4', 'FEAT5', 'FEAT6', 'FEAT7', 'FEAT8', 'FEAT9', 'FEAT10',
             'FEAT11', 'FEAT12', 'FEAT13', 'FEAT14', 'FEAT15', 'FEAT16', 'FEAT17', 'FEAT18', 'FEAT19', 'FEAT20',
             'FEAT21', 'SCORE']
    dataset = pandas.read_csv(url, sep='\t', names=names, skiprows=[0])
    return dataset

def print_data_info(dataset):
    print("##############################################")

    # shape
    print("Shape of dataset:\n")
    print(dataset.shape)
    print("##############################################")

    # head
    print("Head of dataset:\n")
    print(dataset.head(20))
    print("##############################################")

    #descriptions
    print("Description of dataset:\n")
    print(dataset.describe())
    print("##############################################")

    # class distribution
    print("Score distribution for dataset:\n")
    print(dataset.groupby('SCORE').size())
    print("##############################################")

def build_machine_learning():
    get_data()
    dataset = load_data()
    print_data_info(dataset)

