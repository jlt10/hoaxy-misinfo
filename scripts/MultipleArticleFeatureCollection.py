import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.signal import find_peaks
import pytz
from collections import defaultdict

from scripts.MultipleNetworkCollection import collect_tweet_network

data_path = "../data"
features = {
    "num_peaks": 0,
    "active_days": 0,
    "lifespan": 0,
}


def get_relevant_article_ids():
    article_ids = []
    with open(f"{data_path}/2020_article_ids.csv", "r") as f:
        for aid in f.readlines():
            article_ids.append(int(aid.rstrip()))
    return article_ids


def get_filtered_article_dataframe(article_ids):
    article_df = pd.read_csv(f"{data_path}/2020_articles.csv")
    article_df.set_index("id")
    article_df['date_published'] = pd.to_datetime(article_df['date_published'])
    # determine which half of 2020
    article_df['which_half'] = article_df['date_published']
    midway_date = pytz.utc.localize(datetime(2020, 7, 2))
    article_df.loc[article_df['date_published'] < midway_date, 'which_half'] = 1
    article_df.loc[article_df['date_published'] >= midway_date, 'which_half'] = 2
    article_df = article_df[article_df["id"].isin(article_ids)]
    for feature, default in features.items():
        article_df[feature] = default
    article_df.to_csv(f"{data_path}/article_features.csv")
    return article_df


def get_article_features_dataframe():
    article_df = pd.read_csv(f"{data_path}/article_features.csv")
    article_df['date_published'] = pd.to_datetime(article_df['date_published'])
    return article_df


def get_article_tweet_df(aid):
    tweet_df = pd.read_csv(f"../data/networks/{aid}_tweets.csv")
    tweet_df['article_id'] = aid  # set article id
    tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at'])  # convert created_at
    tweet_df["date"] = tweet_df.created_at.dt.date
    return tweet_df


def get_all_article_tweet_dfs(article_ids):
    tweet_df_map = {}
    for aid in article_ids:
        tweet_df_map[aid] = get_article_tweet_df(aid)
    return tweet_df_map


def get_date_range(tweet_df):
    # Include date before the minimum date to catch any initial peaks.
    min_date = tweet_df.created_at.min().date() + timedelta(days=-1)
    max_date = tweet_df.created_at.max().date()
    date_range = np.array([min_date + timedelta(days=x) for x in range((max_date - min_date).days)])
    return date_range


def get_signal_dataframe(tweet_df, date_range):
    id_by_date = tweet_df.groupby("date").count()["id"]
    signal_df = pd.DataFrame({"date": date_range}).join(id_by_date, on="date").fillna(0)
    signal_df.rename(columns={'id': 'tweet_count'}, inplace=True)
    return signal_df


def get_days_active(date_range, signal_df):
    inactive_days = signal_df[signal_df.tweet_count == 0].shape[0]
    return len(date_range) - inactive_days


def get_signal_peaks(aid, x, min_peak=50, graph=True):
    peaks, _ = find_peaks(x, height=min_peak, distance=7)
    if graph:
        save_signal_graph(aid, x, peaks)
    return peaks

# time_slice: amount of samples between peaks, decided in
# in get_signal_peaks()
def get_exp_rates(aid_bursty_map, peaks, signal_df, x, aid, time_slice=7):
    x_data = np.array(range(time_slice+1))
    for peak_index in peaks:
#         date = signal_df['date'][peak_index]
        bursty_metric = 0
        if peak_index - time_slice >= 0:
            ## Add small epsilon.
            y_data = np.array(x[peak_index-time_slice:peak_index+1]) + 1e-7
            log_y_data = np.log(y_data)

            # y = C * e^(rt) => C = e^curve_fit[1], r = e^curve_fit[0]
            curve_fit = np.exp(np.polyfit(x_data, log_y_data, 1))
            ## rise
            bursty_metric += curve_fit[0]

        if peak_index + time_slice+1 < len(x):
            y_data = np.array(x[peak_index:peak_index+time_slice+1]) + 1e-7
            log_y_data = np.log(y_data)

            # y = C * e^(rt) => C = e^curve_fit[1], r = e^curve_fit[0]
            curve_fit = np.exp(np.polyfit(x_data, log_y_data, 1))
            ## decay
            bursty_metric += curve_fit[0]
#         print(date, bursty_metric)
        aid_bursty_map[aid].append(bursty_metric)
    return aid_bursty_map

def remove_n_outliers(n, arr):
    arr.sort()
    return arr[-n:]

def save_rates_graph(aid_bursty_map):
    fact_y, claim_y = [], []
    for aid in aid_bursty_map:
        val = sum(aid_bursty_map[aid])/ len(aid_bursty_map[aid])
        if val == 0 or val == 0.0:
            continue
        article_network_df, _ = collect_tweet_network([aid])
        if article_network_df['site_type'][0] == 'claim':
            claim_y.append(val)
        elif article_network_df['site_type'][0] == 'fact_checking':
            fact_y.append(val)
    ## Remove 3 outliers, these 3 heavily skew the graph to the extent that fact checking data is compressed to one line
    claim_y = remove_n_outliers(3, claim_y)
    plt.clf()
    fig, ax = plt.subplots()
    plt.title(" Article Type vs Contrived metric of burstiness.")

    boxplot_dict = {'Claim': claim_y, 'Fact Checking': fact_y}
    ax.boxplot(boxplot_dict.values())
    ax.set_xticklabels(boxplot_dict.keys())

    plt.xlabel(" Article Type ")
    plt.ylabel(" Contrived metric ")
    plt.savefig("Spread of burstiness (Fano Factor)")
    plt.clf()


def save_signal_graph(aid, x, peaks):
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.title(aid)
    plt.savefig(f"{data_path}/peak_graphs/{aid}_peaks.png")
    plt.clf()


def get_peak_distance(peaks):
    n = len(peaks)
    max_dist = 0
    avg_dist = 0
    for i in range(n - 1):
        d = (peaks[i + 1] - peaks[i])
        avg_dist += d / (n - 1)
        max_dist = max(max_dist, d)
    return avg_dist, max_dist


def get_fano_factor(signal):
    return np.var(signal) / np.mean(signal, dtype=np.float64)


def get_peak_features(article_ids, article_df, tweet_df_map):
    # Maps used to create new columns later.
    all_peaks = []
    dr_col_map = {}
    active_col_map = {}
    npeaks_col_map = {}
    avg_peak_dist_col_map = {}
    max_peak_dist_col_map = {}
    # aid_bursty_map = defaultdict(list)
    fano_factor_col_map = {}

    for aid in article_ids:
        tdf = tweet_df_map[aid]
        dr = get_date_range(tdf)
        signal_df = get_signal_dataframe(tdf, dr)
        x = signal_df.tweet_count.to_numpy()
        fano = get_fano_factor(x)
        peaks = get_signal_peaks(aid, x, graph=True)
        # # Map articles to "burstiness".
        # aid_bursty_map = get_exp_rates(aid_bursty_map, peaks, signal_df, x, aid)
        # Save relevant peak data separately.
        all_peaks.extend([(aid, dr[p], x[p]) for p in peaks])
        # Record relevant lifespan features
        dr_col_map[aid] = len(dr) - 1   # Subtract extra day
        active_col_map[aid] = get_days_active(dr, signal_df)
        npeaks_col_map[aid] = len(peaks)
        fano_factor_col_map[aid] = fano
        # Record peak distance features
        avg_dist, max_dist = get_peak_distance(peaks)
        avg_peak_dist_col_map[aid] = avg_dist
        max_peak_dist_col_map[aid] = max_dist
    # # Draw and save bursty chart
    # save_rates_graph(aid_bursty_map)
    # Save the collection of peaks found.
    all_peaks = np.transpose(all_peaks)
    pd.DataFrame({
        "aid": all_peaks[0],
        "peak_date": all_peaks[1],
        "peak_height": all_peaks[2],
    }).to_csv(f"{data_path}/article_peaks.csv")
    # Save the articles
    article_df["lifespan"] = article_df['id'].map(dr_col_map)
    article_df["active_days"] = article_df['id'].map(active_col_map)
    article_df["num_peaks"] = article_df['id'].map(npeaks_col_map)
    article_df["active_ratio"] = article_df["active_days"] / article_df["lifespan"]
    article_df["fano_factor"] = article_df['id'].map(fano_factor_col_map)
    article_df["avg_peak_dist"] = article_df['id'].map(avg_peak_dist_col_map)
    article_df["max_peak_dist"] = article_df['id'].map(max_peak_dist_col_map)



def main():
    article_ids = get_relevant_article_ids()
    article_df = get_filtered_article_dataframe(article_ids)
    tweet_df_map = get_all_article_tweet_dfs(article_ids)
    # Add new features to the article DF
    get_peak_features(article_ids, article_df, tweet_df_map)
    # Save the new features.
    article_df.to_csv(f"{data_path}/article_features.csv")


if __name__ == "__main__":
    main()
