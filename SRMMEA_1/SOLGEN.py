import random
import numpy as np
from clustering.clusternew import Kmeans_clu
from Mod_Kmeans import Kmeans_clu
from numpy import genfromtxt, linalg, apply_along_axis
from population_creation.population import pop_create
from cluster_validity_indices.metric import cal_pbm_index, silhouette_score

def ClusterIndicesComp(clustNum, labels_array):  # list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])



def FindLabel(x,cluster1,pop,plab):   #Finding the label of x
    for i in range(0,cluster1):
        plot2 = pop[ClusterIndicesComp(i, plab)]
        plot2=plot2[0]
        k=np.where(plot2==x)
        if (len(k[0]!=0)):
            return i
        else:
            print("Not in this label\n")


def solgen(plab,pcenter,pop,cluster1,max_gen): #Initialization
    beta=np.ones(max_gen)
    print("Value of pop is in solgen\n", pop)
    print("Value of beta is \n",beta)
    for i in range(0,len(pop)):
       x=pop[i]
       print("Value of x is in solgen\n",x)
       NP=FindLabel(x,cluster1,pop,plab)  #To find the label where x is
       print("Value of label is\n",NP)
       Sol(x,NP,pop,beta,plab,i)


def Sol(x,NP,pop,beta,plab,i):      #Check if exploration or explotation is to be done
    if (random.uniform(0,1)>beta[0]):
        mat_pool=NP
        F=0.6
        CR=0.9
        p1 = pop[ClusterIndicesComp(NP, plab)]
        p1=random.choice(p1[0])
        p2=random.choice(p1[0])

        DE(p1,p2,x,CR,F)

    else:
        mat_pool=pop
        F = 0.5
        CR = 0.5
        NP=random.choice(plab)
        print("Value of NP is\n",NP)
        p = pop[ClusterIndicesComp(NP, plab)]
        print("Value of population P is \n",p)
        p1 = random.choice(p)
        p2 = random.choice(p)
        print("Value of p1\n", p1)
        print("Value of p2\n", p2)
        DE(p1, p2, x,CR,F,i,pop)


def DE(p1, p2, x, CR, F, i, pop):  # Generate the offspring
    s=[]
    s=np.zeros(len(x))
    for j in range(len(x)):
        if ((random.uniform(0, 1) < CR) or (j == len(x))):
            s[j] = x[j] + F * (p1[j] - p2[j])
        else:
            s[j] = x[j]

    print("Value of S in solgen \n",s)
    Repair(pop, s, i, x)  #Repair the gene



def Repair(pop,s,i,x):     # Checking if it is going beyond the boundary
    Newlab = []  ##Store the labels in lab
    Newcenter = []
    new_center = []
    new_pop = []
    pbm_index_newpop=[]
    sil_score_newpop=[]
    C=[]

    pop=np.array(pop)
    A=pop.max(axis=0) #Getting the maximum from the column
    B=pop.min(axis=0) #Getting the minimum from the column
    for j in range(0,len(s)):
        if (s[j]<A[i]):
           s[j]=A[i]
        elif(s[j]>B[i]):
           s[j]=B[i]
        else:
           s[j]=s[j]


    print("Value of offspring for the {} is {} for axis{}\n".format(x, s, i))
    center1 = s[0:4]
    #print("Value of center 1 is\n", center1)
    center2 = s[4:8]
    #print("Value of center 2 is\n", center2)
    center3 = s[8:12]
    #print("Value of center 3 is\n", center3)
    C.append(center1)
    C.append(center2)
    C.append(center3)
    C = np.array(C)
    print("Value of Initial Center\n",C)
    data = genfromtxt('iris.csv', delimiter=',', skip_header=0, usecols=range(0, 4))  ##Read the input data
    actual_label = genfromtxt('iris.csv', delimiter=',', dtype=str, skip_header=0, usecols=(4))
    u, lab = Kmeans_clu(3, data, C)
    Newlab.insert(i, lab)  #Store the labels in lab
    Newcenter.insert(i, u)
    features = len(data[0])
    max_cluster = 3
    cluster = 3
    new_pbm = cal_pbm_index(cluster, data, u, lab)
    pbm_index_newpop.insert(i, new_pbm)  #Store the second objective function(PBM Ind1ex)
    s = silhouette_score(data, lab)
    sil_score_newpop.insert(i, s)  #Store the first objective funvtion(Silhouette Score)
    new_center = pop_create(max_cluster, features, cluster, u)
    new_pop.insert(i, new_center)











