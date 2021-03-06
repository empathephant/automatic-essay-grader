import os
import csv

def clean_data():
    os.mkdir("./data/30_min_essays")
    os.mkdir("./data/10_min_essays")

    essay_filename = os.path.join("./data", "Raw_Essay_Data.csv")
    essay_data = csv.reader(open(essay_filename, newline='', encoding='latin-1'), quotechar='"')

    for row_num, row in enumerate(essay_data):
        if not row_num == 0:
            new30_filename = f'30-{row[1]}-{float(row[4]):.2f}'
            with open(os.path.join("./data/30_min_essays", new30_filename), "w+") as f:
                f.write(row[3])

            new10_filename = f'10-{row[1]}-{float(row[4]):.2f}'
            with open(os.path.join("./data/10_min_essays", new10_filename), "w+") as f:
                f.write(row[2])
