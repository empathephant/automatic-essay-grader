import os

filename = "Raw_Essay_Data.csv"

with open(filename) as f:
    student_data = f.readlines()

print(student_data)