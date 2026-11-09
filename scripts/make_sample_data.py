"""Generate a SYNTHETIC demonstration dataset with the same columns as the real one.

This file is ONLY for testing that the software works. It is not real X data and
its accuracy must not be reported in the thesis. Replace it with the real labelled
dataset (1,469 genuine and 1,329 fake accounts) before presenting results.

Usage:  python scripts/make_sample_data.py
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def accounts(n, fake):
    # Each account is drawn from a "typical" or an "atypical" profile so that the
    # two classes overlap, as they do in real data.
    atypical = rng.random(n) < 0.12
    def pick(gen_typ, gen_atyp):
        return np.where(atypical, gen_atyp, gen_typ)
    if fake:
        statuses = pick(rng.lognormal(3.5, 1.2, n), rng.lognormal(7.0, 1.0, n))
        followers = pick(rng.lognormal(2.0, 1.3, n), rng.lognormal(5.5, 1.1, n))
        friends = pick(rng.lognormal(6.3, 0.9, n), rng.lognormal(5.6, 1.0, n))
        favourites = pick(rng.lognormal(1.2, 1.5, n), rng.lognormal(5.5, 1.3, n))
        listed = pick(rng.poisson(0.3, n), rng.poisson(6, n))
        url = pick(rng.beta(6, 2, n), rng.beta(2, 5, n))
        mentions = pick(rng.gamma(1.0, 0.15, n), rng.gamma(2.0, 0.3, n))
        hashtags = pick(rng.gamma(1.0, 0.2, n), rng.gamma(2.0, 0.3, n))
        favs = pick(rng.gamma(0.6, 0.15, n), rng.gamma(2.0, 0.4, n))
        rts = pick(rng.gamma(3.0, 0.25, n), rng.gamma(1.5, 0.15, n))
        reply = pick(rng.gamma(0.6, 0.08, n), rng.gamma(2.0, 0.2, n))
    else:
        statuses = pick(rng.lognormal(7.5, 1.1, n), rng.lognormal(4.0, 1.2, n))
        followers = pick(rng.lognormal(5.8, 1.2, n), rng.lognormal(3.0, 1.2, n))
        friends = pick(rng.lognormal(5.6, 1.0, n), rng.lognormal(6.0, 1.0, n))
        favourites = pick(rng.lognormal(6.0, 1.4, n), rng.lognormal(2.0, 1.5, n))
        listed = pick(rng.poisson(8, n), rng.poisson(0.5, n))
        url = pick(rng.beta(2, 6, n), rng.beta(5, 3, n))
        mentions = pick(rng.gamma(2.5, 0.3, n), rng.gamma(1.0, 0.2, n))
        hashtags = pick(rng.gamma(2.0, 0.3, n), rng.gamma(1.0, 0.2, n))
        favs = pick(rng.gamma(2.0, 0.5, n), rng.gamma(0.8, 0.2, n))
        rts = pick(rng.gamma(1.5, 0.15, n), rng.gamma(2.5, 0.25, n))
        reply = pick(rng.gamma(2.0, 0.2, n), rng.gamma(0.8, 0.1, n))
    df = pd.DataFrame({
        "statuses_count": statuses.astype(int), "followers_count": followers.astype(int),
        "friends_count": friends.astype(int), "favourites_count": favourites.astype(int),
        "listed_count": listed.astype(int),
    })
    df["FFratio"] = (df["followers_count"] / df["friends_count"].clip(lower=1)).round(4)
    df["URLratio"] = np.clip(url, 0, 1).round(4)
    df["average_mentions"] = mentions.round(4)
    df["average_hashtags"] = hashtags.round(4)
    df["average_favorites"] = favs.round(4)
    df["average_retweet"] = rts.round(4)
    df["average_reply"] = reply.round(4)
    df["label"] = int(fake)
    return df


if __name__ == "__main__":
    data = pd.concat([accounts(1469, False), accounts(1329, True)]).sample(frac=1, random_state=1)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "sample_profiles_DEMO.csv")
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} synthetic rows to {os.path.abspath(out)}")
