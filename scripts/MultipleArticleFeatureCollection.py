import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.signal import find_peaks
import pytz

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


def save_signal_graph(aid, x, peaks):
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.title(aid)
    plt.savefig(f"{data_path}/peak_graphs/{aid}_peaks.png")
    plt.clf()


def get_peak_features(article_ids, article_df, tweet_df_map):
    # Maps used to create new columns later.
    all_peaks = []
    dr_col_map = {}
    active_col_map = {}
    npeaks_col_map = {}
    for aid in article_ids:
        tdf = tweet_df_map[aid]
        dr = get_date_range(tdf)
        signal_df = get_signal_dataframe(tdf, dr)
        x = signal_df.tweet_count.to_numpy()
        peaks = get_signal_peaks(aid, x, graph=True)
        # Record the relevant features
        dr_col_map[aid] = len(dr) - 1   # Subtract extra day
        active_col_map[aid] = get_days_active(dr, signal_df)
        npeaks_col_map[aid] = len(peaks)
        all_peaks.extend([(aid, dr[p], x[p]) for p in peaks])
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
