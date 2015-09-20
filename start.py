import re
import praw
import sklearn
import filecache
import hand_picked
import from_reddit
import classifiers

# data_source = hand_picked
data_source = from_reddit

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

gender_to_number = {
  'male'   : 1,
  'female' : 0
}



def data_for(label_set):
  X_data = []
  y_data = []

  if type(label_set) == dict:
    iterator = label_set.iteritems()
  elif type(label_set) == list:
    iterator = label_set

  for username, gender in iterator:
    words = get_words_for_redditor(username)
    X_data.append(words)
    y_data.append(gender_to_number[gender])
  return (X_data, y_data)

def data_train():
  return data_for(data_source.training_labels())


def data_test():
  return data_for(data_source.test_labels())
  # return data_for(test_data())

def build_vectorizer():
  # TODO: Provide other vectorizers, given some parameters
  vectorizer = TfidfVectorizer(sublinear_tf=True,
                               max_df=0.5,
                               stop_words='english')
  return vectorizer

def get_reddit_client():
  user_agent = "Karma breakdown 1.0 by /u/_Daimon_"
  return praw.Reddit(user_agent=user_agent)

def strip_markdown_links(text):
  return re.sub('\[.*?\]\(.*?\)', '', text)

@filecache.filecache
def get_words_for_redditor(username):
  print "Grabbing comments for redditor: ", username
  words = u""
  comments = get_reddit_client().get_redditor(username).get_comments()
  for comment in comments:
    words += comment.body + u" "
  # TODO: Strip more out of words. Finding lots of useless artifacts when
  # reviewing the redditor comments.
  return strip_markdown_links(words)

def build_vectorizer():
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    return vectorizer

def benchmark_classifier(classifier, X_train_data, y_train, X_test_data, y_test):
  vectorizer = build_vectorizer()

  X_train = vectorizer.fit_transform(X_train_data)
  classifier.fit(X_train, y_train)

  X_test  = vectorizer.transform(X_test_data)
  prediction = classifier.predict(X_test)

  score = metrics.accuracy_score(y_test, prediction)
  return score

def try_classifiers():
  X_train_data, y_train = data_train()
  print "Training size: ", len(y_train)

  X_test_data, y_test = data_test()
  print "Test size: ", len(y_test)

  for name, classifier in classifiers.with_labels:
    score = benchmark_classifier(classifier,
                                  X_train_data, y_train,
                                  X_test_data, y_test)
    print(name, ' score: ', score)

if __name__ == '__main__':
  try_classifiers()
