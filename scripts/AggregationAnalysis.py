import pandas as pd
import matplotlib.pyplot as plt
from MultipleNetworkCollection import collect_tweet_network
from scipy.signal import find_peaks
import numpy as np


def overlay_by_col(df, col_name, group_by):
    unique_col_vals = df[col_name].unique()
    num_unique = len(unique_col_vals)
    unique_dfs = [None] * num_unique

    # separate data
    for i in range(num_unique):
        unique_dfs[i] = df.loc[df[col_name] == unique_col_vals[i]]

    # plot data
    for i in range(num_unique):
        grouped_df = unique_dfs[i].groupby([group_by]).count()['id']
    #     grouped_df.plot(figsize=(12, 8), stacked=True)
    #
    # # save plot
    # plt.savefig('../plots/tweet_vol_by_' + col_name + '_overlay.pdf')


# day: 0
# hour: 1
# minute: 2
# second: 3
# month: 4

# assumes date col already given
def aggregate_by(df, factor_by=0):
    df_cols = set(df.columns.values.tolist())

    if 'date' not in df_cols:
        df["date"] = df.created_at.dt.date
    factor_cols = ['date', 'type']

    if factor_by == 1:
        if 'hour' not in df_cols:
            df["hour"] = df.created_at.dt.hour
        factor_cols.append('hour')
    elif factor_by == 2:
        if 'minute' not in df_cols:
            df["minute"] = df.created_at.dt.minute
        factor_cols.append('minute')
    elif factor_by == 3:
        if 'second' not in df_cols:
            df["second"] = df.created_at.dt.second
        factor_cols.append('second')
    elif factor_by == 4:
        if 'month' not in df_cols:
            df["month"] = df.created_at.dt.month
        factor_cols.append('month')

    # grouped_df = df.groupby(factor_cols).count()["id"].unstack("type").fillna(0)
    # grouped_df.plot.bar(title="Tweet Volume over Time", figsize=(12, 8), stacked=True)
    #
    # # save plot
    # plt.savefig('../plots/aggregated_by_plot.pdf')


# read 2020 data
article_ids = []
with open(f"../data/2020_article_ids.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append(int(aid.rstrip('\n')))

# print(article_ids)

# article_df = collect_tweet_network(article_ids)

# collect network
article_network_df, article_network_map = collect_tweet_network(article_ids)

article_network_df["date"] = article_network_df.created_at.apply(lambda dt: dt.date)

# overlay_by_col(article_network_df, 'which_half', 'date')


# peaks for all 2020 data
claim_df = article_network_df.loc[article_network_df['site_type'] == 'claim']
fact_df = article_network_df.loc[article_network_df['site_type'] == 'fact_checking']

claim_date_freq = claim_df['date'].value_counts().sort_index()
fact_date_freq = fact_df['date'].value_counts().sort_index()

claim_date_freq_np = claim_date_freq.to_numpy()
claim_date_freq_dates = claim_date_freq.index
fact_date_freq_np = fact_date_freq.to_numpy()
fact_date_freq_dates = fact_date_freq.index

claim_peaks, _ = find_peaks(claim_date_freq_np, height=500, distance=7)
fact_peaks, _ = find_peaks(fact_date_freq_np, height=150, distance=7)


claim_new_dates = []
fact_new_dates = []

for i in range(len(claim_date_freq_np)):
    if i in claim_peaks:
        claim_new_dates.append(str(claim_date_freq_dates[i]))
    else:
        claim_new_dates.append('')

plt.plot(claim_date_freq_np)
plt.plot(claim_peaks, claim_date_freq_np[claim_peaks], "x")
plt.plot(np.zeros_like(claim_date_freq_np), "--", color="gray")
plt.title("Peaks of Tweet Volume from All 2020 Claim Articles")
plt.xlabel("Time by Day")
plt.xticks(np.arange(len(claim_date_freq_dates)), claim_new_dates, rotation='vertical', fontsize=5)
plt.yticks(fontsize=5)
plt.ylabel("Volume of Tweets")
plt.savefig('../plots/agg_peaks_dates_claim.pdf')
plt.clf()


for i in range(len(fact_date_freq_np)):
    if i in fact_peaks:
        fact_new_dates.append(str(fact_date_freq_dates[i]))
    else:
        fact_new_dates.append('')

plt.plot(fact_date_freq_np)
plt.plot(fact_peaks, fact_date_freq_np[fact_peaks], "x")
plt.plot(np.zeros_like(fact_date_freq_np), "--", color="gray")
plt.title("Peaks of Tweet Volume from All 2020 Fact-Checking Articles")
plt.xticks(np.arange(len(fact_date_freq_dates)), fact_new_dates, rotation='vertical', fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel("Time by Day", fontsize=12)
plt.ylabel("Volume of Tweets")
plt.ylim(0, 1000)
plt.savefig('../plots/agg_peaks_dates_fact.pdf')
plt.clf()