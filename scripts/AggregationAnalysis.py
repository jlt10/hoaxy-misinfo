import matplotlib.pyplot as plt


def overlay_by_col(df, col_name):
    unique_col_vals = df[col_name].unique()
    num_unique = len(unique_col_vals)
    unique_dfs = [None] * num_unique

    # separate data
    for i in range(num_unique):
        unique_dfs[i] = df.loc[df[col_name] == unique_col_vals[i]]

    # plot data
    for i in range(num_unique):
        grouped_df = unique_dfs[i].groupby(['date']).count()['id']
        grouped_df.plot(figsize=(12, 8), stacked=True)

    # save plot
    plt.savefig('../plots/tweet_vol_by_' + col_name + '_overlay.pdf')


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

    grouped_df = df.groupby(factor_cols).count()["id"].unstack("type").fillna(0)
    grouped_df.plot.bar(title="Tweet Volume over Time", figsize=(12, 8), stacked=True)

    # save plot
    plt.savefig('../plots/aggregated_by_plot.pdf')
