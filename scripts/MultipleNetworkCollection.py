import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# histograph of volume of tweets in time buckets
article_ids = []
with open(f"../data/article_ids.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append(int(aid.rstrip('\n')))
        
# read in all article info for later reference
all_article_info = pd.read_csv('../data/all_articles.csv')

# function to collect tweet network
def collect_tweet_network(article_ids):
    tweet_df_map = {}
    
    for aid in article_ids:
        tweet_df_map[aid] = pd.read_csv(f"../data/networks/{aid}_tweets.csv")
        
        # article id
        tweet_df_map[aid]['article_id'] = aid
        
        # created at date
        tweet_df_map[aid]['created_at']= pd.to_datetime(tweet_df_map[aid]['created_at'])
        
        # domain url
        tweet_df_map[aid]['domain'] = all_article_info.loc[all_article_info['id'] == int(aid)]['domain'].values[0]
       
    concat_df = pd.concat(tweet_df_map.values())
    return concat_df, tweet_df_map

# collect network
all_network_df, all_network_map = collect_tweet_network(article_ids)

# more info columns
# extract date
all_network_df["date"] = all_network_df.created_at.apply(lambda dt: dt.date)


## plotting
# type of tweet
grouped_all_network = all_network_df.groupby(["date", "type"]).count()["id"].unstack("type").fillna(0)
grouped_all_network.plot.bar(title="Tweet Volume over Time by Type", figsize=(12,8), stacked=True)
plt.savefig('../plots/tweet_vol_by_type.pdf')

# domain
grouped_all_network = all_network_df.groupby(["date", "domain"]).count()["id"].unstack("domain").fillna(0)
grouped_all_network.plot.bar(title="Tweet Volume over Time by Domain", figsize=(12,8), stacked=True)
plt.savefig('../plots/tweet_vol_by_domain.pdf')

# closer look at middle peak
# dates = sorted(all_network_df['date'].to_list())
# print(dates[len(dates)//2])
middle_peak = all_network_df.loc[all_network_df['created_at'] > '20200730']
middle_peak = middle_peak.loc[middle_peak['created_at'] < '20200820']
grouped_middle_peak = middle_peak.groupby(["date", "type"]).count()["id"].unstack("type").fillna(0)
grouped_middle_peak.plot.bar(title="Tweet Volume over Time by Type (Middle Peak)", figsize=(12,8), stacked=True)
plt.savefig('../plots/middle_peak_by_type.pdf')


# scatter plot
# features: domain, date, number of tweets, lifetime of article via tweets, max peak height, number of peaks, plateaus, coefficients
# all_network_df.plot(x='date', y='domain', style=".", Title="Tweet Scatter by Domain")
# plt.savefig('all_network_scatter.pdf')
