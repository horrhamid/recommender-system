from lightfm import LightFM
from lightfm.datasets import fetch_movielens
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lightfm.evaluation import auc_score

movie_lens = fetch_movielens(min_rating=5.0)
print(repr(movie_lens['train']))
print(repr(movie_lens['test']))

train, test = movie_lens['train'], movie_lens['test']

alpha = 1e-3
epochs = 70

adagrad_model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adagrad',
                        user_alpha=alpha,
                        item_alpha=alpha)

adadelta_model = LightFM(no_components=30,
                        loss='warp',
                        learning_schedule='adadelta',
                        user_alpha=alpha,
                        item_alpha=alpha)

adagrad_auc = []
for epochs in range(epochs):
    adagrad_model.fit_partial(train, epochs=1)
    adagrad_auc.append(auc_score(adagrad_model, test).mean())


adadelta_auc = []
for epochs in range(epochs):
    adadelta_model.fit_partial(train, epochs=1)
    adadelta_auc.append(auc_score(adadelta_model, test).mean())

x = np.arange(len(adagrad_auc))
y = np.arange(len(adadelta_auc))
plt.plot(x, np.array(adagrad_auc))
plt.plot(y, np.array(adadelta_auc))
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()
