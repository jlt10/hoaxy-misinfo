import pandas as pd
import matplotlib.pyplot as plt
from MultipleNetworkCollection import collect_tweet_network
from scipy.signal import find_peaks
import numpy as np

def get_fano_factor(signal):
    return  np.var(signal) / np.mean(signal, dtype=np.float64)

def remove_n_outliers(n, arr):
    arr.sort() 
    return arr[-n:] 

# read 2020 data
article_ids = []
with open(f"../data/2020_article_ids.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append([int(aid.rstrip('\n'))])
        
## collect network
claim_freq = []
fact_freq = []

## separate by type
for aid in article_ids:
    article_network_df, _ = collect_tweet_network(aid)
    article_network_df["date"] = article_network_df.created_at.apply(lambda dt: dt.date)
    if article_network_df['site_type'][0] == 'claim':
        claim_freq.extend(article_network_df['date'].value_counts().sort_index().to_numpy())
    elif article_network_df['site_type'][0] == 'fact_checking':
        fact_freq.extend(article_network_df['date'].value_counts().sort_index().to_numpy())


# check aggregate values        
print("Correct claim agg: ", get_fano_factor(claim_freq))
print("Correct fact agg: ", get_fano_factor(fact_freq))

        
print()
print()

claim_x, claim_y , fact_x, fact_y = [], [], [], []

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
## rmove outliers from claim_y
claim_y = remove_n_outliers(3, claim_y)



# print(claim_x, claim_y)

# ## Save id vs burstiness
# ## Claim
# plt.clf()
# plt.title("Claim articles (id) vs burstiness")
# plt.scatter(claim_x, claim_y)
# plt.savefig("Claim articles (id) vs burstiness")
# plt.clf()


# ## Fact-checking
# plt.title("Fact-checking articles (id) vs burstiness")
# plt.scatter(fact_x, fact_y)
# plt.savefig("Fact-checking articles (id) vs burstiness")
# plt.clf()


## All articles
## change to box and whisker plot
# plt.clf()
# plt.title("Claim articles (id) vs burstiness")
# plt.boxplot(claim_y)
# plt.xlabel("Fano Factor")
# # plt.legend()
# plt.savefig("Claim articles (id) vs burstiness (Fano Factor)")
# plt.clf()

plt.clf()
fig, ax = plt.subplots()
plt.title(" Fact Checking Article vs Fano factor")


boxplot_dict = {'Fact Checking': fact_y}
ax.boxplot(boxplot_dict.values())
ax.set_xticklabels(boxplot_dict.keys())

plt.xlabel("Article Type ")
plt.ylabel("Fano factor")
plt.savefig("Both articles: Spread of burstiness (Fano Factor)")




plt.clf()
fig, ax = plt.subplots()
plt.title(" Article Type vs Fano factor")


boxplot_dict = {'Claim': claim_y, 'Fact Checking': fact_y}
ax.boxplot(boxplot_dict.values())
ax.set_xticklabels(boxplot_dict.keys())

plt.xlabel("Article Type ")
#plt.legend([p1, p2], ["Claim Articles", 'Fact checking Articlesgit '])
plt.ylabel("Fano factor")
plt.savefig("Fact Checking: Spread of burstiness (Fano Factor)")
plt.clf()

