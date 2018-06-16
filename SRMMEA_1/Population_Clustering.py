import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd
from clustering.clusternew import Kmeans_clu
import random
from collections import Counter, defaultdict
from math import sqrt
import matplotlib.pyplot as plt
from SOLGEN import solgen
from cluster_validity_indices.metric import cal_pbm_index, silhouette_score

sil_score1 = []
pbm_index1= []
K1 = []
A1 = []
beta = []
p = []
mat_pool = []
X = []
S = []


def ClusterIndicesComp(clustNum, labels_array):  # For extracting  points from the label
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


def Plot(pop,ucenter,ulabel):  #Plotting the graph
    colors=["g.","r.","y.","b.","c."]
    for i in range(len(pop)):
        #print("Cor-ordinates are",pop[i],"label:",ulabel[i])
        plt.plot(pop[i][0],pop[i][1],colors[ulabel[i]],markersize=10)
        plt.scatter(ucenter[:,0],ucenter[:,1],linewidths=5,s=150,zorder=10)
        plt.show()

def index(lst, obj, n):   # To find the index of of Lable which needs to be change
    """

    :param lst: List to be passed (plab)
    :param obj: Which is the object or value to be changed
    :param n: at which index does the object occur
    :return: Index to be return at which the object occurs
    """
    count = 0
    for index, item in enumerate(lst):
        if item == obj:
            count += 1
        if count == n:
            return index
    raise ValueError('{} is not in list at least {} times'.format(obj, n))





def UpdateCentetr_Label(Index,lablelPlot1, plot2, lablePlot2, pop, plab, plot1, pcenter,cluster1,max_gen):
    #print("Value in UpdateCentetLabel of point to be added to cluster\n", plot2)
    #print("Value Label in updated center is\n", plab)
    #print("Value of Label in which Plot2 is\n", lablePlot2)
    #print("Value of Label in which Plot1 is\n", lablelPlot1)

    s = list(plab) #Storing Value of Label in list
    print("Value of  in Label before\n", s)  # Before Updating the value of label
    s1 = Counter(s)
    print("Value of counter in before is \n", s1)
    Index=Index+1
    q = index(s, lablePlot2, Index)  #Storing the index of label which need to be changed
    s[q] = lablelPlot1          #Changing the label by labelPlot1
    ulabel = np.array(s)
    print("Value of an updated label is \n", ulabel)
    print("------------------------------------------")
    # print("Value of Pcenter in Updatedlabel\n",pcenter)
    ncenter = plot2 + plot1 / 2  #Calculating the center of label which have one point associated
    print("Value of new center to be updated\n", ncenter)
    ucenter = np.array(pcenter)
    #print("Value of ucenter is\n",ucenter)
    ucenter[lablelPlot1] = ncenter   #Changing the value of label with new center
    point2 = pop[ClusterIndicesComp(lablePlot2, ulabel)] #Extracting the point of updated label whoes center to be changed
    print("Value where we have to change\n",point2)
    plotk = np.zeros(12)   #Initalizing the with zero value to store the center of LabelPlot2
    k = np.array(len(plot2))
    for i in range(len(plot2)):  # To get one element at a time from plot2
        plotk += plot2[i]

    #print("Value of plotk after addition\n", plotk)
    plotk = np.divide(plotk, k)    #Value of center of Label which need to change
    print("Value of  center is of plot2{}\n".format(plotk))
    ucenter[lablePlot2]=plotk  #Updating the value of label
    print("Value of updated center is\n", ucenter)
    ko = Counter(ulabel)
    print("Value of counter updated label is\n", ko)
    LC1 = [t for (t, v) in ko.items() if v == 1]  # Again checking if there is label which have point associated
    t1 = np.array(LC1)
    if (t1.size):
        One_LengthCluster(cluster1, ulabel, ucenter, pop,max_gen) #To update the it again
    else:
        Plot(pop, ucenter, ulabel)  #To plot the on the graph
        solgen(ulabel,ucenter,pop,cluster1,max_gen)  #Go for the DE operation


    ucenter=[]



def ClusterPopulation(max_gen, population, data):

    """

    :param max_gen:
    :param population:
    :param data:
    :return:
    """

    pop = np.array(population)
    plab1 = []
    pcenter = []
    cluster1 = int(input("Enter the cluster for new population\n"))


    for i in range(0,max_gen):
        if (i % cluster1 == 1):      # Checking the condition of Kmeans Clustering
            K1.insert(i, cluster1)
            u, label, t, l = Kmeans_clu(cluster1, population)   # Storing the values Center(u) anb label(u)
            plab1.insert(i, label)
            pcenter.insert(i, u)
            plab = np.array(plab1)
            plab=plab[0]
            pcenter=np.array(pcenter)
            pcenter=pcenter[0]
            print('Value of label after Clustering of population\n', plab)
            One_LengthCluster(cluster1, plab, pcenter, pop,max_gen)   #To check if any label has one point associated
        else:
            print("Not need of clustering for this generation of\n", i)




def One_LengthCluster(cluster1, plab, pcenter, pop,max_gen):
    indexes = []
    Index=[]   #Store the index of Point which have minimum euclidean distance
    D=[] #Store the  minimum euclidean distance
    labelplot2=[] #Store the Label which have more than 2 points associated
    Point2=[]  #Store the Point which have minimum euclidean distance
    z=[]
    Smin=[]
    I=[]
    L=[]
    LC = Counter(plab)      #Counting number of points associated with label
    print("VAlue of LAbel and Number Cluster Associated with them\n", LC)
    LC1 = [t for (t, v) in LC.items() if v == 1]
    t1 = np.array(LC1)
    if (t1.size):# To check if any of the Label has one point associated
        for b in range(len(t1)):
            plot1 = pop[ClusterIndicesComp(t1[b], plab)]  # Extracting the point in the label which have one point associated
            print("Point of label one Length PLOT1 is\n", np.array(plot1), t1[b])
            z1 = [t for (t, v) in LC.items() if v > 2]  # To check distance with label which more than 3 points associated
            z = np.array(z1)   #Storing the value in the array
            for d in range(len(z)):
                print("Value of Label which have more than two cluster is\n", z[d])
                plot2 = pop[ClusterIndicesComp(z[d], plab)] # Extracting the point in the label more than one point associated
                print("Value of plot2 in one length cluster is\n", plot2)
                for i in range(len(plot2)):
                    plotk = plot2[i]    # To get one point at a time from plot2
                    S = np.linalg.norm(np.array(plot1) - np.array(plotk))
                    print("Distance between {} and {} is {}\n".format(plot1,plotk,S))  # euclidian distance is calculated
                    if (i == 0):
                        Smin = S
                        Sminant = S
                        indexes.append(i)
                    else:
                        if (S < Sminant):
                            Smin = S
                            Sminant = Smin
                            indexes = []
                            indexes.append(i)
                        elif (S == Sminant):
                            indexes = []
                            indexes.append(i)

                #print('indexes:')
                print("Index at which the minimum value is stored\n", indexes)  # To find the index of Label with which euclidian distance is minimum

                for i in range(len(indexes)):
                    Point2 = plot2[indexes[i]]
                    I = indexes[i]
                    L = z[d]
                    print("VAlues of Point{} which have min distance with plot1 is in Label {} and have Index {} and distance {}\n".format(Point2,L,I,Smin))

                if(len(z)==1):  #If Label which have more than 2 point associated is only one
                    D = Smin
                    Index = indexes[i]
                    labelplot2=z[d]
                    Point2=plot2[indexes[i]]
                    print("Here is the value\n", D, Index, labelplot2, Point2)
                    UpdateCentetr_Label(Index,t1[b], Point2, z[d], pop, plab, plot1, pcenter,cluster1,max_gen)  #After Finding Point now update center and label
                    break


                elif (len(z) > 1):  #If Label which have more than 2 point associated is more than one
                    D.append(Smin)
                    Index.append(I)
                    labelplot2.append(L)
                    #print("Value in list are------------\n", labelplot2)
                print("Index value is\n",Index)
                print("Label value is\n", labelplot2)

            z=min(D)    #Finding the minimum distance among all the labels
            k=D.index(z)  #Finding the index where minimum distance is stored in D
            Index=Index[k]
            labelplot2=labelplot2[k]
            Point2 = pop[ClusterIndicesComp(labelplot2, plab)]
            Point2=Point2[Index]
            print("Value of minimum distance is\n",z,Index,labelplot2,k,Point2)
            UpdateCentetr_Label(Index,t1[b], Point2, labelplot2, pop, plab, plot1, pcenter,cluster1,max_gen)  #After Finding Point now update center and label


        D=[]
        indexes=[]

    else:               #If no solution have one point associated in the label
        #Plot(pop, ucenter, ulab)
        solgen(plab,pcenter,pop,cluster1,max_gen)
        print("no lenght 1 cluster\n")


