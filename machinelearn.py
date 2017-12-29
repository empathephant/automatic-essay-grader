import os

import features

def get_data():
    if not os.path.isfile("feature_data.tsv"):
        features.get_feature_data()

def build_machine_learning():
    get_data()
