import os

import nltk
from nltk.corpus import cmudict
prondict = cmudict.dict()

import cleandata

def get_essay_data():
    if not os.path.exists("./data/30_min_essays") and not os.path.exists("./data/30_min_essays"):
        print("Cleaning data...")
        cleandata.clean_data()
        print("Cleaning complete.")
    else:
        print("Data already cleaned.")

def feat1(lines):
    # Returns rate of lines that begin with a lowercase letter
    numTimes = 0
    for line in lines:
        if len(line) > 0 and line[0].islower():
            numTimes += 1
    if len(lines) == 0:
        return -1
    else:
        normedRate = (numTimes / len(lines))
        return normedRate

def feat2(sentences):
    # Returns rate of sentences that begin with a lowercase letter
    numTimes = 0
    for sent in sentences:
        if len(sent) > 0 and sent[0].islower():
            numTimes += 1
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (numTimes / len(sentences))
        return normedRate

def feat3(pos_tagged, sentences):
    # Returns average number of verbs per sentence
    numTimes = 0
    for tok, tag in pos_tagged:
        if tag[:1] == 'V':
            numTimes += 1
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (numTimes / len(sentences))
        return normedRate

def feat4(tokens, sentences):
    # Returns average number of words per sentence
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (len(tokens) / len(sentences))
        return normedRate

def feat5(tokens):
    # Returns average number of letters per word
    numChars = 0
    for tok in tokens:
            numChars += len(tok)
    if len(tokens) == 0:
        return -1
    else:
        normedRate = (numChars / len(tokens))
        return normedRate

def feat6(sentences):
    # Returns total sentence count
    return len(sentences)

def feat7(tokens):
    # Returns total word count
    return len(tokens)

def feat8(pos_tagged, sentences):
    # Returns average number of adjectives per sentence
    numTimes = 0
    for tok, tag in pos_tagged:
        if tag[:1] == 'J':
            numTimes += 1
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (numTimes / len(sentences))
        return normedRate

def feat9(pos_tagged, sentences):
    # Returns average number of adverbs per sentence
    numTimes = 0
    for tok, tag in pos_tagged:
        if tag[:1] == 'R':
            numTimes += 1
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (numTimes / len(sentences))
        return normedRate

def feat10():
    # Returns Flesch-Kincaid readability
    return 3

def feat11():
    # Returns Flesch-Kincaid grade level
    return 3

def feat12():
    # Number of hapaxes
    return 3

def feat13():
    # Number of SAT lemmas / lemma
    return 3

def feat14():
    # Unique lemmas / token ratio
    return 3

def feat15(pos_tagged):
    # Returns total number of foreign words
    numTimes = 0
    for tok, tag in pos_tagged:
        if tag[:1] == 'F':
            numTimes += 1
    if len(pos_tagged) == 0:
        return -1
    else:
        return numTimes

def feat16(pos_tagged, sentences):
    # Returns average number of modals per sentence
    numTimes = 0
    for tok, tag in pos_tagged:
        if tag[:1] == 'M':
            numTimes += 1
    if len(sentences) == 0:
        return -1
    else:
        normedRate = (numTimes / len(sentences))
        return normedRate

def feat17():
    # Returns Gunning fox index
    return 3

def feat18():
    # Returns ARI (Automated readability index)
    return 3

def feat19():
    # Returns Fry readability score
    return 3

def feat20():
    # Returns SMOG grade
    return 3

def feat21():
    # Returns Coleman–Liau index
    return 3

def write_feature_data():
    with open(os.path.join('./data', "feature_data.tsv"), "w+") as tsvfile:
        tsvfile.write("IDNUM\t")
        for i in range (1, 22):
            tsvfile.write(f'FEAT{i}\t')
        tsvfile.write("SCORE\n")

        for essay_filename in os.listdir("./data/30_min_essays"):
            if essay_filename[0] == "3":
                f = open(os.path.join('./data/30_min_essays', essay_filename), "r")
                raw_text = f.read()

                tokens = nltk.word_tokenize(raw_text)
                text = nltk.Text(tokens)
                lines = raw_text.splitlines()
                sentences = nltk.sent_tokenize(raw_text)
                pos_tagged = nltk.pos_tag(tokens)

                student_id = essay_filename[3:8]
                score = float(essay_filename[9:13])

                tsvfile.write(f'{student_id}\t')
                tsvfile.write(f'{feat1(lines):.4f}\t')
                tsvfile.write(f'{feat2(sentences):.4f}\t')
                tsvfile.write(f'{feat3(pos_tagged, sentences):.4f}\t')
                tsvfile.write(f'{feat4(tokens, sentences):.4f}\t')
                tsvfile.write(f'{feat5(tokens):.4f}\t')
                tsvfile.write(f'{feat6(sentences):.4f}\t')
                tsvfile.write(f'{feat7(tokens):.4f}\t')
                tsvfile.write(f'{feat8(pos_tagged, sentences):.4f}\t')
                tsvfile.write(f'{feat9(pos_tagged, sentences):.4f}\t')
                tsvfile.write(f'{feat10():.4f}\t')
                tsvfile.write(f'{feat11():.4f}\t')
                tsvfile.write(f'{feat12():.4f}\t')
                tsvfile.write(f'{feat13():.4f}\t')
                tsvfile.write(f'{feat14():.4f}\t')
                tsvfile.write(f'{feat15(pos_tagged):.4f}\t')
                tsvfile.write(f'{feat16(pos_tagged, sentences):.4f}\t')
                tsvfile.write(f'{feat17():.4f}\t')
                tsvfile.write(f'{feat18():.4f}\t')
                tsvfile.write(f'{feat19():.4f}\t')
                tsvfile.write(f'{feat20():.4f}\t')
                tsvfile.write(f'{feat21():.4f}\t')
                tsvfile.write(f'{int(round(score, 0))}\n')

        print("Analysis finished.")

def get_feature_data():
    get_essay_data()
    write_feature_data()
