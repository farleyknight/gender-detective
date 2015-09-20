import re
import praw
import filecache

def contains(patterns, string):
  if type(patterns) != list:
    patterns = [patterns]
  return any([len(re.findall(p, string)) > 0 for p in patterns])


def flair_to_gender(flair):
  if flair == None or flair == '':
    return None

  # Since female contains the word male, check for
  # that first
  if contains(['female', u'\u2640'], flair.lower()):
    return 'female'

  if contains(['male', 'dude', u'\u2642'], flair.lower()):
    return 'male'

  # print 'Couldn\'t figure out flair: ', flair
  return None

def genders_from_flair(subreddit):
  comments = comments_from_subreddit(subreddit)
  genders = []
  for i, c in enumerate(comments):
    genders.append((c.author.name, flair_to_gender(c.author_flair_text)))
  return filter(lambda x: x[1] != None, genders)


@filecache.filecache
def comments_from_subreddit(subreddit):
  print "Getting comments from subreddit", subreddit
  r = get_reddit_client()
  sub = r.get_subreddit(subreddit)
  comments = []
  for i, comment in enumerate(sub.get_comments(limit=None)):
    print 'HTTP GET for comment', i
    comments.append(comment)
  return comments

def get_reddit_client():
  user_agent = "Karma breakdown 1.0 by /u/_Daimon_"
  return praw.Reddit(user_agent=user_agent)


def find_gender_set():
  ask_women_genders = genders_from_flair('AskWomen')
  ask_men_genders   = genders_from_flair('AskMen')
  gender_set        = list(set(ask_men_genders + ask_women_genders))
  return (gender_set, int(len(gender_set) * 0.90))

def training_labels():
  gender_set, ninety_percent = find_gender_set()
  return gender_set[0:ninety_percent]

def test_labels():
  gender_set, ninety_percent = find_gender_set()
  return gender_set[ninety_percent:len(gender_set)]
