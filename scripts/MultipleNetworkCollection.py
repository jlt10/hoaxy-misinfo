import pandas as pd

# histograph of volume of tweets in time buckets
article_ids = []
with open(f"../data/article_ids.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append(aid)

def collect_tweet_network(article_ids): 
	tweet_dfs = []
	for aid in article_ids: 
		tweet_df = pd.read_csv(f"../data/networks/{aid}_tweets.csv")
		tweet_df['article_id'] = aid
		tweet_df['created_at']= pd.to_datetime(tweet_df['created_at'])
		tweet_dfs.append(tweet_df)
	concat_df = pd.concat(tweet_dfs)
	return concat_df