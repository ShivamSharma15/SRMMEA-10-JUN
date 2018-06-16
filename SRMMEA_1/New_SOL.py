import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd
from clustering.clusternew import Kmeans_clu
from SR import solgen
import random
from collections import Counter, defaultdict
from math import sqrt
import matplotlib.pyplot as plt

K1 = []
A1 = []

beta = []
p = []
mat_pool = []
X = []
S = []


def ClusterIndicesComp(clustNum, labels_array):  # list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


# def ChangeLabel(l,l1,pl2,l2,data):
#   print("Value of pl2",pl2)
# k = list(np.where(l.labels_ == l1))
# k=data[ClusterIndicesComp(l1,l.labels_)]
# print("Value of k is ",k)
# c = np.r_[k[None, :], pl2[None, :]]
# print("Value of z is\n",z)
# k=list(np.where(l.labels_==l2))
# k.pop(pl2)

def Plot(pop, ucenter, ulabel):
    colors = ["g.", "r.", "y.", "b.", "c."]
    for i in range(len(pop)):
        print("Cor-ordinates are", pop[i], "label:", ulabel[i])
        plt.plot(pop[i][0], pop[i][1], colors[ulabel[i]], markersize=10)
        plt.scatter(ucenter[:, 0], ucenter[:, 1], marker="X", linewidths=5, s=150, zorder=10)
        plt.show()


def UpdateCentetr_Label(lablelPlot1, plot2, lablePlot2, pop, plab, plot1, pcenter):
    print("Value in UpdateCentetLabel of point to be added to cluster\n", plot2)
    print("Value of Length 1 cluster in UpdatedCenre\n", plab)
    print("Value of Label in which Plot2 is\n", lablePlot2)
    print("Value of Label in which Plot1 is\n", lablelPlot1)

    s = list(plab[0])
    print("Value of s in kCenre before\n", s)
    s1 = Counter(s)
    print("Value of counter in Updated is \n", s1)
    if ([t for (t, v) in s1.items() if v == 1]):
        y = s.index(lablePlot2)
        s[y] = lablelPlot1
        ulabel = np.array(s)
        print("Value of an updated label after having length  is 1 \n", ulabel)
        ko = Counter(ulabel)
        print("Value of counter updated label is\n", ko)
    elif ([t for (t, v) in s1.items() if v > 1]):
        ulabel = np.array(s)
        print("Value of an updated label after no length  is 1 \n", ulabel)
        ko = Counter(ulabel)
        print("Value of counter updated label is\n", ko)

    print("------------------------------------------")
    # print("Value of Pcenter in Updatedlabel\n",pcenter)
    ncenter = plot2 + plot1 / 2
    print("Value of new center to be updated\n", ncenter)
    ucenter = np.array(pcenter)
    x = np.where(ucenter == plot1)
    ucenter[x] = ncenter
    print("Value of updated center is\n", ucenter)
    Plot(pop, ucenter, ulabel)
    print("Value of list Center is\n", x)

    ki = x.index(plot1)
    x[ki] = ncenter
    ucenter = np.array(x)


# print("VAlue of updated center is\n",ucenter)
plab = []
pcenter = []
ucenter = []


# print("Value of center in Updatedlabel\n", center)


#    return ulabel,ucenter

def ClusterPopulation(max_gen, population, data):
    # print("Value of population",population)
    pop = np.array(population)
    # print("Data type of population",pop.dtype)
    # print("Population is in array form\n",pop)
    plab = []
    pcenter = []
    cluster1 = int(input("Enter the cluster for new population\n"))

    for i in range(max_gen):
        if (i % cluster1 == 1):
            print("CLustering will occur", i)
            K1.insert(i, cluster1)
            print('value of K1', K1)
            u, label, t, l = Kmeans_clu(cluster1, population)
            A1.insert(i, t)
            plab.insert(i, label)
            pcenter.insert(i, u)
            # print('Value of newcenter\n',pcenter)
            print('Value of new label after Clustering of population\n', plab)
            LC = Counter(l.labels_)
            print("VAlue of LAbel and Number Cluster Associated with them\n", LC)
            LC1 = [t for (t, v) in LC.items() if v == 1]
            t1 = np.array(LC1)
            if (t1.size):
                One_LengthCluster(t1, cluster1, plab, pcenter, l, LC, pop)
            else:
                print("no lenght 1 cluster\n")
                return (plab, pcenter)
        else:
            print("Not need of clustering\n", i)


def One_LengthCluster(t1, cluster1, plab, pcenter, l, LC, pop):
    Slist = []
    indexes = []

    for j in range(1, cluster1):
        for b in range(len(t1)):
            print("Value in NEW_SOL is of 1 length cluster\n", t1[b])
            plot1 = pop[ClusterIndicesComp(t1[b], l.labels_)]  # LEngth ONE FEATURES
            print("Values are in sol of plot1 one Length\n", plot1, t1[b])
            z1 = [t for (t, v) in LC.items() if v > 2]
            z = np.array(z1)
            print("VAlue of cluster in more than 2", z)
            for d in range(len(z)):
                print("Value in NEW_SOL is of more than 2 length cluster\n", z[d])
                plot2 = pop[ClusterIndicesComp(z[d], l.labels_)]
                for i in range(len(plot2)):  # To get one element at a time from plot2
                    plotk = plot2[i]
                    S = np.linalg.norm(np.array(plot1) - np.array(plotk))
                    print("Distance between plot1 and plotk is %f" % (S))  # euclidian distance is calculated
                    if (i == 0):
                        Smin = S
                        Sminant = S
                        indexes.append(i)
                    else:
                        if (S < Sminant):
                            Smin = S
                            indexes = []
                            indexes.append(i)
                        elif (S == Sminant):
                            indexes = []
                            indexes.append(i)

                print('indexes:')
                print(indexes)

                for i in range(len(indexes)):
                    print("VAlues of Slist with min  \n", indexes[i], plot2[indexes[i]], Smin)
                    UpdateCentetr_Label(t1[b], plot2[indexes[i]], z[d], pop, plab, plot1, pcenter)
        indexes = []

        # print("I am here\n")
        # plab=[]
        # pcenter=[]

        # print("Value of min is\n",Smin)
        # print("VAlue of plotk is\n",i,plotk )
        # numpy_calc_dist(plot1, plotk)
        # print("Result is in S",S)

    # for i in range(max_gen):
    # beta=np.ones(1,dtype=int)
    # print('Value of intial probability',beta)
    # for i in range(l.n_clusters):
    # k = np.where(l.labels_ == i)
    # x1=len(l.n_clusters==i)
    # print("Length of cluster of length using old method\n",x1)
    # print("For the cluster values are in New_SOL in K\n", i, k)

    # solgen(5,population,*A1)






