import re
import argparse
from tqdm import tqdm
import pandas as pd

# Default output file
DEFAULT_OUTPUT = "data/csv/tweets.csv"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Read and convert the tweets text file into structured dataset.")
parser.add_argument("input", help="Text file containing tweets")
parser.add_argument("output", nargs='?', default=DEFAULT_OUTPUT,
                    help="File to save tweet's dataset (%s by default)" % DEFAULT_OUTPUT)
args = parser.parse_args()


def process_data(data):
    """ Read and process text file containing tweets.
        Return pandas DataFrame of tweets.
    """
    label_pat = '\n([A-Z][a-zA-Z\_\-]+\ \:\ )'
    start = 'TextTW : '
    data = data.split(start)[1:]
    tweets = []
    for d in tqdm(data):
        tweet = {}
        labels = re.findall(label_pat, d)
        labels.insert(0, start)
        if 'Tweetid : ' in labels:
            if labels.index('Tweetid : ') != 1:
                labels.remove(labels[1])        # for catch all TextTW
        if 'Tweetid : ' not in labels:		    # tweet bad formatted
            continue
        for i in range(len(labels)):
            if 0 <= i < (len(labels) - 1):
                split_word = labels[i+1]
                key = labels[i].replace(' : ', '').replace('-', '_').lower().title()
                content = re.sub('(\s)+$', '', d.split(split_word)[0].replace('\n', ' '))
                tweet[key] = content.lower()
                d = d.split(split_word)[1]
            elif i == (len(labels)-1):
                split_word = labels[i]
                key = labels[i].replace(' : ', '').replace('-', '_').lower().title()
                content = re.sub('(\s)+$', '', d.split(split_word)[0].replace('\n', ' '))
                tweet[key] = content.lower()
                d = ""
        tweets.append(tweet)

    return tweets


def find_city(place):
    city_re = re.compile(r"u\'name': u\'([\w\s]+)\'")
    city = city_re.findall(str(place))
    if city:
        return city[0]
    else:
        return None


def retweet2tweet(tweets):
    rt_pat = "^rt\s@[\w\W]+:\s"
    original_tweets = []
    for text, idx in tqdm(tweets[tweets.Retweet].groupby("Texttw").groups.items()):
        # select last record
        tweet = tweets.loc[idx].fillna('').to_dict(orient="records")[-1]
        # adjust text
        tweet["Texttw"] = re.sub(rt_pat, "", text)
        # adjust other features
        keys = list(tweet.keys())
        for k in keys:
            if "_Author" in k:
                if tweet[k] != '':
                    tweet[k.replace("_Author", "")] = tweet[k]
        original_tweets.append(tweet)
    # remove re-tweets
    tweets.drop(tweets[tweets.Retweet].index, inplace=True)
    re_tweets = pd.DataFrame(original_tweets).fillna('')
    print(re_tweets[re_tweets.Created_At == ''])
    tweets = tweets.append(pd.DataFrame(original_tweets))
    # remove useless columns
    tweets.drop([c for c in tweets.columns if "_Author" in c], axis=1, inplace=True)
    tweets.set_index("Tweetid", inplace=True)
    return tweets


if __name__ == '__main__':
    data = open(args.input, "r").read()

    print("Analyze input file...")
    tweets = process_data(data)

    tweets = pd.DataFrame(tweets)
    tweets.drop_duplicates("Tweetid", keep="last", inplace=True)
    tweets.fillna('')
    tweets["Place"] = tweets["Place"].apply(find_city)
    tweets["Retweet"] = tweets["Texttw"].apply(lambda x: x.startswith("rt"))

    print("Convert useless re-tweets in original tweets...")
    tweets = retweet2tweet(tweets)

    tweets.to_csv(args.output)
    print("Dataset saved in %s" % args.output)
