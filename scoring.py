from nltk.sentiment.vader import SentimentIntensityAnalyzer as SA
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
import argparse
import pandas as pd
import emot
import re
from tqdm import tqdm

# Default output
DEFAULT_OUTPUT = "data/csv/score.csv"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Compute score of each tweet using different methods")
parser.add_argument("input", help="XML file containing tokenized tweets")
parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                    help="CSV file to save score of each tweet (%s by default)" % DEFAULT_OUTPUT)
# parser.add_argument("-m", "--mode", type=str, required=True, choices=["vader", "swn", "mix"],
#                     help="Methods to be used to calculate the scores")
args = parser.parse_args()


def compute_emojis_score(emojis):
    """ Calculate score for each emoji.
        Each emoji is traduced in text and then text score is calculated (Vader).
    """
    emojis_scores = 0.0
    if not emojis:
        return emojis_scores
    for emoji in emojis:
        try:
            emojis_scores += SA().polarity_scores(
                re.sub('[^A-Za-z]', ' ', emot.UNICODE_EMO[emoji])
            )['compound']
        except KeyError:
            emojis_scores += 0.0
    emojis_scores /= len(emojis)
    return emojis_scores


def compute_emoticons_score(emoticons):
    """ Calculate score for each emoticons.
        Each emoticon score is calculated with Vader
    """
    emoticons_scores = 0.0
    if not emoticons:
        return emoticons_scores
    for emoticon in emoticons:
        emoticons_scores += SA().polarity_scores(emoticon)['compound']
    emoticons_scores /= len(emoticons)
    return emoticons_scores


def swn_score(text):
    """ Calculate score with sentiwordnet library.
        Return score for sentence.
    """
    score = 0.0

    if text is not None:
        # mark negation
        words = mark_negation(text.split())

        # remove stopwords
        words = [t for t in words if t not in stopwords.words('english')]

        # select sense for each word
        words_sense = {}
        for word in words:
            clean_word = word.replace('_NEG', '')
            if wn.synsets(clean_word):
                words_sense[word] = wn.synsets(clean_word)[0]

        # calculate score
        for word, sense in words_sense.items():
            pos_score = swn.senti_synset(sense.name()).pos_score()
            neg_score = swn.senti_synset(sense.name()).neg_score()
            if '_NEG' in word:
                pos_score, neg_score = neg_score, pos_score
            score += (pos_score - neg_score)
        if len(words_sense) != 0:
            score /= len(words_sense)
    return score


def vader_score(tweet):
    """ Calculate score with vader library.
        Return score for sentence.
    """
    # text = tweet["Stemmed_Text"]
    text = tweet["Plain_Text"]
    emojis = tweet["Emojis"].split()
    score = SA().polarity_scores(text)['compound']
    if len(emojis) != 0:
        emojis_score = 0.0
        for emoji in emojis:
            try:
                emojis_score += SA().polarity_scores(
                    re.sub('[^A-Za-z]', ' ', emot.UNICODE_EMO[emoji])
                )['compound']
            except KeyError:
                emojis_score += 0.0
        score += emojis_score
        score /= len(emojis) + 1  # (1 for text)
    return score


def mix_score(tweet):
    # text = tweet["Stemmed_Text"]
    text = tweet["Plain_Text"]
    emojis = tweet["Emojis"].split()
    emoticons = tweet["Emoticons"].split()
    score = swn_score(text)
    if len(emojis) != 0:
        emoijs_score = compute_emojis_score(emojis)
        score += emoijs_score
        score /= 2
    if len(emoticons) != 0:
        emoticons_score = compute_emoticons_score(emoticons)
        score += emoticons_score
        score /= 2
    return score


if __name__ == '__main__':
    tweets = pd.read_csv(args.input, index_col=0).fillna('')
    scores = pd.DataFrame()
    tqdm.pandas()

    scores["DATETIME"] = pd.to_datetime(tweets["Created_At"])
    # scores["SWN"] = tweets["Stemmed_Text"].progress_apply(swn_score)
    # scores["VADER"] = tweets[["Stemmed_Text", "Emojis"]].progress_apply(vader_score, axis=1)
    # scores["MIX"] = tweets[["Stemmed_Text", "Emojis", "Emoticons"]].progress_apply(mix_score, axis=1)

    print("Compute SWN score...")
    scores["SWN"] = tweets["Plain_Text"].progress_apply(swn_score)
    print("Compute vader score...")
    scores["VADER"] = tweets[["Plain_Text", "Emojis"]].progress_apply(vader_score, axis=1)
    print("Compute mix score...")
    scores["MIX"] = tweets[["Plain_Text", "Emojis", "Emoticons"]].progress_apply(mix_score, axis=1)

    scores.to_csv(args.output)
    print("Results saved in %s" % args.output)



