
# TODO

## Easy TODO:

* Collect over 200 male & female combined usernames (DONE)
* Training set and test set should be randomized from one master set
* Collect more comments from each user
* Collect self-post titles & content from users
* Collect post titles from users, not necessarily just self-post titles
* Try bigram / trigrame bag of words
* Analyze / categorize comments by subreddit

## Moderate (somewhat demanding) TODO

* Collect subreddit submissions from user, use this as a second classifier and
  combine this classifier with the original text document classifier
* Do part-of-speech tagging and create classifier just on nouns, verbs,
  adjectives, etc.

## Hard (more demanding) TODO

* Scrub the data really, really well. Remove unicode characters. Fix mispellings.
  Possibly look into stemming.
* Use NLP to find statements like 'As a woman', 'As a guy', 'my husband',
  'my wife', etc.
* Try fitting word2vec somewhere in here.
