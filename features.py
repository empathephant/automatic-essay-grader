import os

import nltk

import cleandata

def get_essay_data():
    if not os.path.exists("./data/30_min_essays") and not os.path.exists("./data/30_min_essays"):
        print("Cleaning data.")
        cleandata.clean_data()
    else:
        print("Data already cleaned.")

def feat1():
    return 1

def feat2():
    return 2

def feat3():
    return 3

def write_feature_data():
    with open(os.path.join('./data', "feature_data.tsv"), "w+") as tsvfile:
        tsvfile.write("IDNUM\tSCORE\tFEAT1\tFEAT2\tFEAT3\n")

        for essay_filename in os.listdir("./data/30_min_essays"):
            f = open(os.path.join('./data/30_min_essays', essay_filename), "r")
            raw_text = f.read()

            tokens = nltk.word_tokenize(raw_text)
            text = nltk.Text(tokens)
            lines = raw_text.splitlines()
            sentences = nltk.sent_tokenize(raw_text)

            student_id = essay_filename[3:8]
            score = essay_filename[9:13]

            tsvfile.write(f'{student_id}\t')
            tsvfile.write(f'{score}\t')
            tsvfile.write(f'{feat1():.4f}\t')
            tsvfile.write(f'{feat2():.4f}\t')
            tsvfile.write(f'{feat3():.4f}\n')

        print("Analysis finished.")

def get_feature_data():
    get_essay_data()
    write_feature_data()
