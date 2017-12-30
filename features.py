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

def feat1(essay_text):
    # Returns rate of paragraphs that begin with a lowercase letter
    numTimes = 0
    for line in essay_text['paragraphs']:
        if len(line) > 0 and line[0].islower():
            numTimes += 1
    if essay_text['para_count'] == 0:
        return -1
    else:
        normedRate = (numTimes / essay_text['para_count'])
        return normedRate

def feat2(essay_text):
    # Returns rate of sentences that begin with a lowercase letter
    numTimes = 0
    for sent in essay_text['sentences']:
        if len(sent) > 0 and sent[0].islower():
            numTimes += 1
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate = (numTimes / essay_text['sent_count'])
        return normedRate

def feat3(essay_text):
    # Returns average number of verbs per sentence
    numTimes = 0
    for tok, tag in essay_text['pos_tagged']:
        if tag[:1] == 'V':
            numTimes += 1
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate = (numTimes / essay_text['sent_count'])
        return normedRate

def feat4(essay_text):
    # Returns average number of words per sentence
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate=(essay_text['word_count'] / essay_text['sent_count'])
        return normedRate

def feat5(essay_text):
    # Returns average number of letters per word
    numChars = 0
    for word in essay_text['words_only']:
            numChars += len(word)
    if essay_text['word_count'] == 0:
        return -1
    else:
        normedRate=(numChars / essay_text['word_count'])
        return normedRate

def feat6(essay_text):
    # Returns total sentence count
    return essay_text['sent_count']

def feat7(essay_text):
    # Returns total word count
    return essay_text['word_count']

def feat8(essay_text):
    # Returns average number of adjectives per sentence
    numTimes = 0
    for tok, tag in essay_text['pos_tagged']:
        if tag[:1] == 'J':
            numTimes += 1
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate=(numTimes / essay_text['sent_count'])
        return normedRate

def feat9(essay_text):
    # Returns average number of adverbs per sentence
    numTimes=0
    for tok, tag in essay_text['pos_tagged']:
        if tag[:1] == 'R':
            numTimes += 1
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate=(numTimes / essay_text['sent_count'])
        return normedRate

def feat10(essay_text):
    # dummy function
    return 3


def feat11(essay_text):
    # dummy function
    return 3

# def feat10(essay_text):
#     # Returns Flesch readability
#     readability = 206.835 - (1.015 * (essay_text['word_count'] / essay_text['sent_count']))
#     readability -= 84.6 * (essay_text['syllable_count'] / essay_text['word_count'])
#     return readability

# def feat11(essay_text):
#     # Returns Flesch-Kincaid grade level
#     grade_level = 0.39 * (essay_text['word_count'] / essay_text['sent_count'])
#     grade_level += (11.8 * (essay_text['syllable_count'] / essay_text['word_count'])) - 15.59
#     return grade_level

def feat12(essay_text):
    # Number of hapaxes
    return 3

def feat13(essay_text):
    # Number of SAT lemmas / lemma
    return 3

def feat14(essay_text):
    # Unique lemmas / token ratio
    return 3

def feat15(essay_text):
    # Returns total number of foreign words
    numTimes = 0
    for tok, tag in essay_text['pos_tagged']:
        if tag[:1] == 'F':
            numTimes += 1
    if essay_text['word_count'] == 0:
        return -1
    else:
        return numTimes

def feat16(essay_text):
    # Returns average number of modals per sentence
    numTimes=0
    for tok, tag in essay_text['pos_tagged']:
        if tag[:1] == 'M':
            numTimes += 1
    if essay_text['sent_count'] == 0:
        return -1
    else:
        normedRate=(numTimes / essay_text['sent_count'])
        return normedRate

def feat17(essay_text):
    # Returns Gunning fox index
    return 3

def feat18(essay_text):
    # Returns ARI (Automated readability index)
    return 3

def feat19(essay_text):
    # Returns Fry readability score
    return 3

def feat20(essay_text):
    # Returns SMOG grade
    return 3

def feat21(essay_text):
    # Returns Coleman–Liau index
    return 3

def not_punct(tok):
    if len(tok) == 1 and not tok.isalpha():
        return False
    else:
        return True

# def numsyllables_pronlist(pronlist):
#     return len(list(filter(lambda segmental: segmental.lower()[-1].isdigit(), pronlist)))
#     # Returns number of vowels marked in the pronunciation dictionary

# def word_syllable_count(word):
#     try:
#         return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
#     except KeyError as e:
#         return [0]

# def total_syllable_count(words_only):
#     sum(map(lambda word: max(word_syllable_count(word)), words_only))

def prepare_text(essay_filename, raw_text):
    essay_text = {}

    essay_text['tokens'] = nltk.word_tokenize(raw_text)
    essay_text['words_only'] = list(filter(not_punct, essay_text['tokens']))
    essay_text['word_count'] = len(essay_text['words_only'])

    essay_text['text'] = nltk.Text(essay_text['tokens'])
    essay_text['pos_tagged'] = nltk.pos_tag(essay_text['tokens'])

    essay_text['paragraphs'] = raw_text.splitlines()
    essay_text['para_count'] = len(essay_text['paragraphs'])
    essay_text['sentences'] = nltk.sent_tokenize(raw_text)
    essay_text['sent_count'] = len(essay_text['sentences'])
    # essay_text['syllable_count'] = total_syllable_count(essay_text['words_only'])

    essay_text['student_id'] = essay_filename[3:8]
    essay_text['score'] = float(essay_filename[9:13])

    return essay_text

def write_feature_data():
    with open(os.path.join('./data', "feature_data.tsv"), "w+") as tsvfile:
        tsvfile.write("IDNUM\t")
        for i in range (1, 22):
            tsvfile.write(f'FEAT{i}\t')
        tsvfile.write("SCORE\n")

        for essay_filename in os.listdir("./data/30_min_essays"):
            if essay_filename[0] == "3":
                with open(os.path.join('./data/30_min_essays', essay_filename), "r") as f:
                    raw_text = f.read()

                essay_text = prepare_text(essay_filename, raw_text)
                print(essay_text['student_id'])

                tsvfile.write(essay_text['student_id'] + '\t')
                tsvfile.write(f'{feat1(essay_text):.4f}\t')
                tsvfile.write(f'{feat2(essay_text):.4f}\t')
                tsvfile.write(f'{feat3(essay_text):.4f}\t')
                tsvfile.write(f'{feat4(essay_text):.4f}\t')
                tsvfile.write(f'{feat5(essay_text):.4f}\t')
                tsvfile.write(f'{feat6(essay_text):.4f}\t')
                tsvfile.write(f'{feat7(essay_text):.4f}\t')
                tsvfile.write(f'{feat8(essay_text):.4f}\t')
                tsvfile.write(f'{feat9(essay_text):.4f}\t')
                tsvfile.write(f'{feat10(essay_text):.4f}\t')
                tsvfile.write(f'{feat11(essay_text):.4f}\t')
                tsvfile.write(f'{feat12(essay_text):.4f}\t')
                tsvfile.write(f'{feat13(essay_text):.4f}\t')
                tsvfile.write(f'{feat14(essay_text):.4f}\t')
                tsvfile.write(f'{feat15(essay_text):.4f}\t')
                tsvfile.write(f'{feat16(essay_text):.4f}\t')
                tsvfile.write(f'{feat17(essay_text):.4f}\t')
                tsvfile.write(f'{feat18(essay_text):.4f}\t')
                tsvfile.write(f'{feat19(essay_text):.4f}\t')
                tsvfile.write(f'{feat20(essay_text):.4f}\t')
                tsvfile.write(f'{feat21(essay_text):.4f}\t')
                tsvfile.write(str(int(round(essay_text['score'], 0))) + '\n')

def get_feature_data():
    get_essay_data()
    write_feature_data()
