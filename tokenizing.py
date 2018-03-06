import re
import html.entities
import emot
from nltk.sentiment.util import HAPPY, SAD
import argparse
import pandas as pd
from tqdm import tqdm

# Default output
DEFAULT_OUTPUT = "data/csv/tkn_tweets.csv"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Tokenize tweets text and save results.")
parser.add_argument("input", help="CSV file containing tweets")
parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                    help="File to save tokenized tweets (%s by default)" % DEFAULT_OUTPUT)
args = parser.parse_args()

# ======= Variable for tokenizing ======= #
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

regex_strings = (
    # Emoticons:
    emoticon_string
    ,
    # HTML tags:
    r"""(?:<[^>]+>)"""
    ,
    # Retweet
    r"""RT"""
    ,
    # URLs:
    r"""(?:http[s]?://t.co/[a-zA-Z0-9]+)"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+:?)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots.
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
)

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)
# ======================================= #


def html2unicode(text):
    """	Convert HTML entities in unicode char.
    """
    html_entity_digit_re = re.compile(r"&#\d+;")
    html_entity_alpha_re = re.compile(r"&\w+;")
    amp = "&amp;"

    # digit
    ents = set(html_entity_digit_re.findall(text))
    if len(ents) > 0:
        for ent in ents:
            entnum = ent[2:-1]
            entnum = int(entnum)
            text = text.replace(ent, chr(entnum))

    # alpha
    ents = set(html_entity_alpha_re.findall(text))
    ents = filter((lambda x: x != amp), ents)
    for ent in ents:
        entname = ent[1:-1]
        text = text.replace(ent, chr(html.entities.name2codepoint[entname]))

    text = text.replace(amp, " and ")

    return text


def find_emojis(text):
    """ Find and remove emojis in text.
        Return emojis founded and text without emojis.
    """
    emojis = []
    for emoji in emot.emoji(text):
        emojis.append(emoji['value'])
        text = text.replace(emoji['value'], '')

    return text, emojis


def split_hashtags(hashtag):
    """ Split camel case hashtag
        Return splitted hashtag
    """
    matches = re.findall('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', hashtag)
    return matches


def find_token(text, kind):
    valid_arg = ["hashtags", "plain", "emojis", "emoticons"]
    if kind not in valid_arg:
        raise ValueError("Invalid argument. Please select from %s" % valid_arg)
    text, emojis = find_emojis(text)
    hashtags = []
    plain_text = []
    emoticons = []
    tokens = word_re.findall(text)
    # find...
    for token in tokens:
        if token.startswith('#'):  # ...hashtag
            hashtags.append(token)
            hashtag_text = split_hashtags(token.replace('#', ''))
            for hashtag in hashtag_text:
                plain_text.append(hashtag)
        elif token.startswith('@'):  # ...screen name (pass)
            pass
        elif token.startswith('http'):  # ...link (pass)
            pass
        elif (token in HAPPY) or (token in SAD):  # ...emoticon
            emoticons.append(token)
            plain_text.append(token)
        else:
            plain_text.append(re.sub('[_\-]', ' ', token).lower())  # ...plain text
    switcher = {
        'hashtags': ' '.join(hashtags),
        'plain': ' '.join(plain_text),
        'emoticons': ' '.join(emoticons),
        'emojis': ' '.join(emojis)
    }
    return switcher.get(kind, None)


if __name__ == '__main__':
    tweets = pd.read_csv(args.input, index_col=0)

    # Convert html entities in unicode char
    print("Tokenizing...")
    tweets["Texttw"] = tweets["Texttw"].apply(html2unicode)
    tweets["Hashtags"] = tweets["Texttw"].apply(find_token, args=("hashtags", ))
    tweets["Plain_Text"] = tweets["Texttw"].apply(find_token, args=("plain", ))
    tweets["Emojis"] = tweets["Texttw"].apply(find_token, args=("emojis", ))
    tweets["Emoticons"] = tweets["Texttw"].apply(find_token, args=("emoticons", ))

    tweets.to_csv(args.output)
    print("Results saved in %s" % args.output)
