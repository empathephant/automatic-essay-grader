# automatic-essay-grader
Machine learning project that attempts to grade essays written by non-native English speakers on a scale from 1 - 7.

## Dependencies

* nltk
* * cmudict
* * averaged_perceptron_tagger
* numpy
* pandas
* matplotlib
* sklearn

## Set-up

Run the file "essay_grader.py"

## Note to Dr. Reynolds

I did, in fact, code all of this myself. Github shows other people making commits because due to the explosion of my former laptop I have had to use others' devices.

## Next Steps

1. Add in more features to test for, such as readability scores
1. Clean up existing features, only using most effective ones to increase accuracy
1. Throw out any rows that return a -1 for a value
1. Add more training data to improve accuracy
1. Figure out how to make the algorithm give a rating rather than a categorical score
1. Make an interface that allows you to grade an essay from the command line
1. Look at 10 minute essay data
