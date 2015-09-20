import re
import praw
import sklearn
import filecache

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



training_labels = {
  # Males Train
  'BlakBanana'          : 'male',   # 1
  'elementality22'      : 'male',   # 2
  'neroli90'            : 'male',   # 3
  'NightSoD'            : 'male',   # 4
  'mad_bad_dangerous'   : 'male',   # 5
  'TWSpriggs'           : 'male',   # 6
  'PhuPhuSnugglyShitz'  : 'male',   # 7
  'DJ-Salinger'         : 'male',   # 8
  'TTRoadHog'           : 'male',   # 9
  'Iama_Kokiri_AMA'     : 'male',   # 10
  'ajustin118'          : 'male',   # 11
  'MCMXChris'           : 'male',   # 12
  'Whatsthedealwithair' : 'male',   # 13
  'SalamanderSylph'     : 'male',   # 14
  'TTRoadHog'           : 'male',   # 15
  'caffiend98'          : 'male',   # 16
  'athmirleen'          : 'male',   # 17
  'acamu5x'             : 'male',   # 18
  '3kris3'              : 'male',   # 19
  'NightSoD'            : 'male',   # 20

  # Female Train
  'ShesGotSauce'        : 'female', # 1
  'Madame-Ovaries'      : 'female', # 2
  'ridiculousrssndoll'  : 'female', # 3
  'airforcehelpplease'  : 'female', # 4
  'SarahlovesChar'      : 'female', # 5
  'projectbadasss'      : 'female', # 6
  'me_tootwo'           : 'female', # 7
  'muki_mono'           : 'female', # 8
  'neenoonee'           : 'female', # 9
  'JamaisVue'           : 'female', # 10
  'puddlesofblood'      : 'female', # 11
  'neroli90'            : 'female', # 12
  'DearLola'            : 'female', # 13
  'sweetheartX0X0'      : 'female', # 14
  'Pizzanomics_'        : 'female', # 15
  'Penguin_Hat'         : 'female', # 16
  'saratonin84'         : 'female', # 17
  'bellewhether'        : 'female', # 18
  'undertheaurora'      : 'female', # 19
  'luckycharmertoo'     : 'female', # 20
}

test_labels = {
  # Male Test
  'Pg21_SubsecD_Pgrph12' : 'male',    # 1
  'Zamoyski'             : 'male',    # 2
  'afrocanadian'         : 'male',    # 3
  'UniversityDaniel'     : 'male',    # 4
  'Needlecrash'          : 'male',    # 5

  # Female Test
  'giantlegume'          : 'female',   # 1
  'arcticfoxtrotter'     : 'female',   # 2
  'sehrah'               : 'female',   # 3
  'bookwench'            : 'female',   # 4
  'samanthais'           : 'female',   # 5
}

gender_to_number = {
  'male'   : 1,
  'female' : 0
}

classifiers = [
  (
    'RidgeClassifier(tol=1e-2, solver="lsqr")',
    RidgeClassifier(tol=1e-2, solver="lsqr")
  ),

  (
    'Perceptron(n_iter=50)',
    Perceptron(n_iter=50)
  ),

  (
    'PassiveAggressiveClassifier(n_iter=50)',
    PassiveAggressiveClassifier(n_iter=50)
  ),

  (
    'KNeighborsClassifier(n_neighbors=10)',
    KNeighborsClassifier(n_neighbors=10)
  ),

  (
    'RandomForestClassifier(n_estimators=100)',
    RandomForestClassifier(n_estimators=100)
  ),

  (
    "LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3)",
    LinearSVC(loss='l2', penalty='l1', dual=False, tol=1e-3)
  ),

  (
    "LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)",
    LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
  ),

  (
    "SGDClassifier(alpha=.0001, n_iter=50, penalty='l1')",
    SGDClassifier(alpha=.0001, n_iter=50, penalty='l1')
  ),

  (
    "SGDClassifier(alpha=.0001, n_iter=50, penalty='l2')",
    SGDClassifier(alpha=.0001, n_iter=50, penalty='l2')
  ),

  (
    'SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")',
    SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
  ),

  (
    'NearestCentroid',
    NearestCentroid()
  ),

  (
    'MultinomialNB(alpha=.01)',
    MultinomialNB(alpha=.01)
  ),

  (
    'BernoulliNB(alpha=.01)',
    BernoulliNB(alpha=.01)
  )
]


def data_for(label_set):
  X_data = []
  y_data = []
  for username, gender in label_set.iteritems():
    words = get_words_for_redditor(username)
    X_data.append(words)
    y_data.append(gender_to_number[gender])
  return (X_data, y_data)

def data_train():
  return data_for(training_labels)
  # return data_for(training_data())

def data_test():
  return data_for(test_labels)
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
  print("Grabbing comments for redditor: ", username)
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

def benchmark_classifier(classifier):
  vectorizer = build_vectorizer()

  X_train_data, y_train = data_train()
  X_train = vectorizer.fit_transform(X_train_data)

  classifier.fit(X_train, y_train)

  X_test_data, y_test = data_test()
  X_test  = vectorizer.transform(X_test_data)

  prediction = classifier.predict(X_test)
  score = metrics.accuracy_score(y_test, prediction)
  return score

def try_classifiers():
  for name, classifier in classifiers:
    score = benchmark_classifier(classifier)
    print(name, ' score: ', score)
