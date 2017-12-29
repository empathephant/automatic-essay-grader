import os

import nltk

import cleandata

def get_essay_data():
    if not os.path.isdir("10_min_essays") and not os.path.isdir("30_min_essays"):
        cleandata.clean_data()

def feat1():
    return 0

def feat2():
    return 0

def feat3():
    return 0

def write_feature_data():
    with open('feature_data.tsv', "w+") as tsvfile:
        tsvfile.write("STUDENT ID\tSCORE\tFEAT1\tFEAT2\tFEAT3\n")


        for essay_filename in os.listdir('30_min_essays'):
            f = open(os.path.join('./30_min_essays', essay_filename), "r")
            raw_text = f.read()

            tokens = nltk.word_tokenize(raw_text)
            text = nltk.Text(tokens)
            lines = raw_text.splitlines()
            sentences = nltk.sent_tokenize(raw_text)

            student_id = essay_filename[3:7]
            score = essay_filename[9:12]

            tsvfile.write(f'{student_id}\t')
            tsvfile.write(f'{score}\t')
            tsvfile.write(f'{feat1():.4f}\t')
            tsvfile.write(f'{feat2():.4f}\t')
            tsvfile.write(f'{feat3():.4f}\n')

        print("Analysis finished.")

def get_feature_data():
    get_essay_data()
    write_feature_data()
