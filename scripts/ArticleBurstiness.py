import pandas as pd
import matplotlib.pyplot as plt
from MultipleNetworkCollection import collect_tweet_network
from scipy.signal import find_peaks
import numpy as np

def get_fano_factor(signal):
    return np.mean(signal, dtype=np.float64) / np.var(signal)

# read 2020 data
article_ids = []
with open(f"../data/2020_article_ids.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append([int(aid.rstrip('\n'))])
        
## collect network
claim_x, claim_y = [], []
fact_x, fact_y = [], []
for aid in article_ids:
    article_network_df, _ = collect_tweet_network(aid)
    article_network_df["date"] = article_network_df.created_at.apply(lambda dt: dt.date)
    article_freq = article_network_df['date'].value_counts().sort_index().to_numpy()
    fano = get_fano_factor(article_freq)
    if article_network_df['site_type'][0] == 'claim':
        claim_x.append(aid[0])
        claim_y.append(fano)
    elif article_network_df['site_type'][0] == 'fact_checking':
        fact_x.append(aid[0])
        fact_y.append(fano)
# print(claim_x, claim_y)

## Save id vs burstiness
## Claim
plt.clf()
plt.title("Claim articles (id) vs burstiness")
plt.scatter(claim_x, claim_y)
plt.savefig("Claim articles (id) vs burstiness")
plt.clf()


## Fact-checking
plt.title("Fact-checking articles (id) vs burstiness")
plt.scatter(fact_x, fact_y)
plt.savefig("Fact-checking articles (id) vs burstiness")
plt.clf()



