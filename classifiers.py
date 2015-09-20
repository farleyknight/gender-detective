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

with_labels = [
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
    "LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3)",
    LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3)
  ),

  (
    "LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3)",
    LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3)
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
    'NearestCentroid()',
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
