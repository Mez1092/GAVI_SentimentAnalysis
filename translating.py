import argparse
import pandas as pd
from googletrans import Translator
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer

DEFAULT_OUTPUT = "data/csv/trsl_tweets.csv"

parser = argparse.ArgumentParser(description="Translate tweets")
parser.add_argument("input", help="Input file containing tweets")
parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                    help="File to save translated text (%s by default)" % DEFAULT_OUTPUT)
args = parser.parse_args()


def translate_txt(tweet, dest_lang='en'):
    src_lang = tweet["Lang"]
    text = tweet["Texttw"]
    if src_lang == 'en':
        return text
    try:
        translated_text = Translator().translate(text, src=src_lang, dest=dest_lang).text
    except ValueError:
        translated_text = text
    return translated_text


def stemming_txt(text):
    stemmer = WordNetLemmatizer()
    stemmed_text = [stemmer.lemmatize(word) for word in text.split()]
    return ' '.join(stemmed_text)


if __name__ == '__main__':
    tweets = pd.read_csv(args.input, index_col=0, nrows=20)
    tweets.fillna("", inplace=True)

    tqdm.pandas()

    tweets["Translated_Text"] = tweets[["Texttw", "Lang"]].progress_apply(translate_txt, axis=1)
    tweets["Stemmed_Text"] = tweets["Translated_Text"].progress_apply(stemming_txt)

    tweets.to_csv(args.output)
    print("Results saved in %s" % args.output)
