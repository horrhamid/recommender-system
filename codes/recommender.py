from lightfm import LightFM
from lightfm.datasets import fetch_movielens
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lightfm.evaluation import auc_score

movie_lens = fetch_movielens(min_rating=5.0)
print(repr(movie_lens['train']))
print(repr(movie_lens['test']))


def sample_recommendation(model, data, user_ids):


    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

train, test = movie_lens['train'], movie_lens['test']

alpha = 1e-3
epochs = 70

model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adagrad',
                        user_alpha=alpha,
                        item_alpha=alpha)

adagrad_auc = []
for epochs in range(epochs):
    model.fit_partial(train, epochs=1)
    adagrad_auc.append(auc_score(model, test).mean())


sample_recommendation(model, movie_lens, [3, 25, 450, 100])

