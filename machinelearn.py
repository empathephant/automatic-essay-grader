import os

import features

def get_data():
    if not os.path.isfile("feature_data.tsv"):
        features.get_feature_data()

get_data()
