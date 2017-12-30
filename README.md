# automatic-essay-grader
Machine learning project that attempts to grade essays written by non-native English speakers on a scale from 1 - 7.

## Dependencies

* nltk
* * cmudict
* * averaged_perceptron_tagger
* * wordnet
* numpy
* pandas
* matplotlib
* sklearn

## Set-up

Run the file "essay_grader.py"

## Note to Dr. Reynolds

I did, in fact, code all of this myself. Github shows other people making commits because due to the explosion of my former laptop I have had to use others' devices.

As you can see, accurcy scores were not great, but it was a fun exercise that I'd like to spend more time on perfecting. Since my time was more limited I tried to make it functional so you could see my understanding of the concepts but I've outlined a lot of the next steps I would have liked to take (and probably will as a personal project!) Please email me if you have any questions. I believe my code is correct in terms of style but may not always be written in the most elegant or efficient way.

## Next Steps

1. Add in more features to test for, such as readability scores suggested in commented out functions
1. Clean up existing features, only using most effective ones to increase accuracy
    * Better understand this error: _"UserWarning: Variables are collinear."_ Is this why accuracy scores went down as I added new features?
1. Throw out any rows that return a -1 for a value
1. Add more training data to improve accuracy
1. Figure out how to make the algorithm give a rating rather than a categorical score
1. Make an interface that allows you to grade an essay from the command line
1. Look at 10 minute essay data
1. Find better way to organize all the feature functions
    * Give feature functions descriptive names and put them in a dictionary or something to loop over
    * Implement `essay_text` as a class instead of a dictionary
1. Learn how to use the settings on the visualizations and make them actually legible
1. Define dependencies better???
1. Investigate warning: "Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples."

## Contributing

Anyone who sees this project and would like to work on adding more features please fork and send a PR! :)
