'''
10/02/20
linear regression prediction model
plotting raw data by category of question
auto cluster numbers applied to plots
loop through csv files for each category (i.e. course ID, ethnicity etc.)
'''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import sklearn
from scipy.spatial.distance import cdist
from sklearn import linear_model
import os



def ml_model():

    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.4)

    # find best model & save as pickle file (only needs to be run once):
    best = 0
    for _ in range(30):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.4)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

        if acc > best:
            best = acc
            with open("overall_score_pred.pickle", "wb") as f:
                pickle.dump(linear, f)

    pickle_in = open("overall_score_pred.pickle", "rb")
    linear = pickle.load(pickle_in)
    #print("Co: ", linear.coef_)
    #print("Intercept: ", linear.intercept_)

    predictions = linear.predict(x_test)

    #for x in range(len(predictions)):
        #print(predictions[x], x_test[x], y_test[x])


def find_best_cluster_number():
    # create new folder to save plots to for each category
    folder_name = os.path.splitext(filename)[0]
    os.makedirs(cwd + "/plots/" + folder_name)

    for i in data:
        X = np.array([data[i], data["Overall satisfaction"]]).transpose()

        # plot elbow
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}

        n_samples = X.shape[0]

        if n_samples >= 10:
            K = range(1, 10)
        else:
            K = range(1, n_samples)

        for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)

            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeanModel.inertia_)

            mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                           'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = kmeanModel.inertia_


        # auto-calculate k (number of clusters)
        error = []
        # delta1[0] must == 0 as delta1[1] = error[0]-error[1]
        delta1 = [0]
        # delta2[0] & [1] must == 0 as delta2[2] = delta1[1]-delta1[2] & delta1[0] always == 0
        delta2 = [0, 0]
        elbow = []
        strength = []
        rel_strength = []
        d1 = distortions[0]

        if n_samples >= 10:
            count = list(range(0, 11))
        else:
            count = list(range(0, n_samples))

        for d in distortions:
            k_error = d / d1
            error.append(k_error)

        for c in count:
            try:
                del_1 = error[c] - error[c + 1]
                delta1.append(del_1)
            except:
                pass

        for c in count:
            try:
                del_2 = delta1[c + 1] - delta1[c + 2]
                delta2.append(del_2)
            except:
                pass

        for c in count:
            try:
                if delta2[c + 1] > delta1[c + 1]:
                    elbow.append(True)
                else:
                    elbow.append(False)
            except:
                pass
        # adds an extra 0 on the end to make data correct length
        elbow.append(0)

        for c in count:
            try:
                if (delta2[c + 1] - delta1[c + 1]) > 0:
                    strength.append(delta2[c + 1] - delta1[c + 1])
                else:
                    strength.append(0)
            except:
                pass
        # adds an extra 0 on the end to make data correct length
        strength.append(0)

        for c in count:
            try:
                if strength[c] > 0:
                    strength[c] / K[c]
                    rel_strength.append(strength[c] / K[c])
                else:
                    rel_strength.append(0)

            except:
                pass

        merged = np.array([K, distortions, error, delta1, delta2, elbow, strength, rel_strength]).transpose()
        c_num = (np.argmax(merged[0:-1, -1])) + 1
        # Show cluster number in output
        #print(c_num)

        kmeans = KMeans(n_clusters=c_num)
        kmeans.fit(X)
        kmeansLabels = kmeans.predict(X)

        # plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=kmeansLabels, s=50, cmap='viridis')
        clusterCenters = kmeans.cluster_centers_
        plt.scatter(clusterCenters[:, 0], clusterCenters[:, 1], c='red', s=200, alpha=0.8)
        plt.xlabel(i)
        plt.ylabel("Overall Satisfaction")
        save_results_to = cwd + "/plots/" + folder_name + "/"
        plt.savefig(save_results_to + i)
        plt.clf()


cwd = os.getcwd()

for filename in os.listdir(cwd + "/csv files"):
    print(filename)
    data = pd.read_csv(cwd + "/csv files/" + filename, sep=",", engine='python')

    data = data[["The teaching on my course", "Learning opportunities",
                 "Assessment and feedback", "Academic support", "Organisation and management",
                 "Learning resources", "Learning community", "Student Voice",
                 "The students union (association or guild) effectively represents students academic interests.",
                 "Overall satisfaction"]]

    predict = "Overall satisfaction"


    ml_model()
    find_best_cluster_number()











