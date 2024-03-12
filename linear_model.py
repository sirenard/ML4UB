import gzip
import pickle

import numpy as np
import pyscipopt
from sklearn.model_selection import KFold

from model import get_sample_files
from sklearn.linear_model import LinearRegression


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def train(data_path, save_path):
    files = get_sample_files(data_path)
    X = []
    Y = []
    qids = []
    index = 0

    for qid, file in enumerate(files):
        with gzip.open(file, "rb") as f:
            f, _, action_set, scores = pickle.load(f)
            features = f.features
            qids.append([])

            scores = scores[action_set]
            scores = scores/np.linalg.norm(scores)
            scores = softmax(scores)
            Y.extend(scores.tolist())

            for action in action_set:
                x = np.nan_to_num(features[action]).astype(float)
                X.append(x)
                #Y.append(scores[action])
                qids[-1].append(index)
                index += 1

    cv = KFold(n_splits=5)

    X = np.array(X)

    pos = np.where(X==0)
    for i,j in zip(*pos):
        X[i,j] = 0.001


    Y = np.array(Y)

    # max_idx = np.where(Y >= 1.05*np.max(Y))[0]
    # Y[:] = 0
    # Y[max_idx] = 1
    #
    # n = Y.size - 2*len(max_idx)
    # indices = np.random.choice(np.where(Y != 0)[0], n)
    # X = np.concatenate((X, X[indices]), axis=0)
    # Y = np.concatenate((Y, Y[indices]), axis=0)


    train, test = cv.split(X, Y).__next__()


    model = LinearRegression()
    model.fit(X[train], Y[train])

    print('score: ', model.score(X[test], Y[test]) )

    accuracy = 0
    y_pred = model.predict(X)
    for ids in qids:
        predicted = y_pred[ids]
        reality = Y[ids]
        accuracy += np.argmax(reality) == np.argmax(predicted)
    accuracy /= len(qids)

    print("Accuracy: ", accuracy)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model


def train2(data_path, save_path):
    files = get_sample_files(data_path)
    X = []
    Y = []
    index = 0

    for qid, file in enumerate(files):
        with gzip.open(file, "rb") as f:
            f, vars_val, lp_obj, obj = pickle.load(f)
            features = np.array(vars_val)

            score = -(lp_obj-obj)/lp_obj

            X.append(features)
            Y.append(score)

    cv = KFold(n_splits=5)

    X = np.array(X)

    pos = np.where(X==0)
    for i,j in zip(*pos):
        X[i,j] = 0.001


    Y = np.array(Y)


    train, test = cv.split(X, Y).__next__()


    model = LinearRegression()
    model.fit(X[train], Y[train])

    print('score: ', model.score(X[test], Y[test]) )

    ys = model.predict(X[test])

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model


def get_limit(model, m: pyscipopt.Model):
    features = []
    for var in m.getVars():
        features.append(var.getLPSol())

    return (1+model.predict(features)) * m.getLPObjVal()

if __name__ == '__main__':
    train2("bounds", "test.pkl")