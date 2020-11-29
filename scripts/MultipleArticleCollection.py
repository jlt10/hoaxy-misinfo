import json
import requests

article_ids = []
with open(f"../data/2020_articles_secondhalf.csv", "r") as f:
    for aid in f.readlines():
        article_ids.append(int(aid.rstrip()))

# Get network of a tweet

url = "https://api-hoaxy.p.rapidapi.com"

headers = {
    'x-rapidapi-key': "6dda23c3b6msh40fb43640818777p173117jsn2e799b5b84a7",
    'x-rapidapi-host': "api-hoaxy.p.rapidapi.com"
}


def fetch_article_tweets(article_id):
    querystring = {"ids": str([article_id])}
    print(querystring)
    response = requests.request(
        "GET", url+"/tweets", headers=headers, params=querystring)
    return json.loads(response.text)


def fetch_article_network(article_id, node_limit=2000, edge_limit=200000, include_mentions=False):
    querystring = {
        "ids": str([article_id]),  # ids must be an array
        "nodes_limit": str(node_limit),
        "edges_limit": str(edge_limit),
        "include_user_mentions": "true" if include_mentions else "false",
    }
    response = requests.request(
        "GET", url+"/network", headers=headers, params=querystring)
    return json.loads(response.text)


class User:
    def __init__(self, user_id, screen_name):
        self.id = user_id
        self.screen_name = screen_name

    def __str__(self):
        return f"User({self.id},{self.screen_name})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, obj):
        return isinstance(obj, User) and obj.id == self.id


class Tweet:
    def __init__(self, tweet_dict, from_edge=False):
        self.id = tweet_dict["tweet_id"]
        self.url = tweet_dict["canonical_url"]
        self.created_at = tweet_dict["tweet_created_at"]
        if from_edge:
            self.is_mention = tweet_dict["is_mention"]
            self.type = tweet_dict["tweet_type"]
            self.url_id = tweet_dict["url_id"]
            user_key = ("from" if self.type == "origin" else "to") + "_user_id"
            self.user_id = tweet_dict[user_key]
        else:
            self.is_mention = None
            self.type = None
            self.url_id = None
            self.user_id = None

    def __str__(self):
        return f"Tweet(id={self.id},created={self.created_at})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, obj):
        return isinstance(obj, Tweet) and obj.id == self.id


class Edge:
    def __init__(self, edge_dict):
        self.url = edge_dict["canonical_url"]
        self.article_id = edge_dict["id"]
        self.from_user = User(
            edge_dict["from_user_id"],
            edge_dict["from_user_screen_name"]
        )
        self.to_user = User(
            edge_dict["to_user_id"],
            edge_dict["to_user_screen_name"]
        )
        self.tweet = Tweet(edge_dict, from_edge=True)

    def __str__(self):
        return f"Edge(tweet={self.tweet.id},from={self.from_user.id},to={self.to_user.id})"


def parse_raw_data(raw_network, raw_tweets):
    # Initialize collections.
    users = dict()
    network = dict()
    tweets = {t["tweet_id"]: Tweet(t) for t in raw_tweets}
    # Iterate through the network.
    for e in raw_network:
        edge = Edge(e)
        tweets[edge.tweet.id] = edge.tweet
        network[edge.tweet.id] = edge
        users[edge.from_user.id] = edge.from_user
        users[edge.to_user.id] = edge.to_user
    # Return the collections.
    return users, network, tweets

# users, network, tweets = parse_raw_data(raw_network, raw_tweets)


network_data_path = "../data/networks/"


def save_user_csv_data(article_id, data_path, users):
    with open(f"{data_path}{article_id}_users.csv", "w") as f:
        f.write("id,screen_name\n")
        for u in users.values():
            f.write(f"{u.id},{u.screen_name}\n")


def save_tweet_csv_data(article_id, data_path, tweets):
    with open(f"{data_path}{article_id}_tweets.csv", "w") as f:
        f.write("id,url_id,user_id,created_at,type,is_mention\n")
        for t in tweets.values():
            f.write(
                f"{t.id},{t.url_id},{t.user_id},{t.created_at},{t.type},{t.is_mention}\n")


def save_network_csv_data(article_id, data_path, network):
    with open(f"{data_path}{article_id}_edges.csv", "w") as f:
        f.write("tweet_id,url,from_user_id,to_user_id\n")
        for e in network.values():
            f.write(f"{e.tweet.id},{e.url},{e.from_user.id},{e.to_user.id}\n")


def save_article_hoaxy_data(article_id, data_path):
    # Fetch tweets and network from API
    raw_tweets = fetch_article_tweets(article_id)["tweets"]
    raw_network = fetch_article_network(article_id)
    if "edges" in raw_network:
        raw_network = raw_network["edges"]
    else:
        raw_network = []
    # Parse raw data into users, networks, and tweets
    users, network, tweets = parse_raw_data(raw_network, raw_tweets)
    # Save all the data
    save_user_csv_data(article_id, data_path, users)
    save_tweet_csv_data(article_id, data_path, tweets)
    save_network_csv_data(article_id, data_path, network)


for aid in article_ids: 
	save_article_hoaxy_data(aid, network_data_path)
