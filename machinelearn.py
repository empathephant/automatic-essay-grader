import os
from pathlib import Path

import features

def get_data():
    tsv_path = Path("./data/feature_data.tsv")

    if not tsv_path.is_file():
        features.get_feature_data()

def build_machine_learning():
    get_data()
