import numpy as np
import random
from evolution.generate import mutate
from Non_DominatedSort import nd_sort
def nsga_3(*population):
    st=[]
    F1=[]
    i=1
    Qt=mutate(a, b, y2, feat)
    Rt=population.union(Qt)
    for i in Qt:
        O, F  = Kmeans_clu(cluster, data)
        F1.insert(i,F)
    while(len(st)>=len(population)):
        st=st.union(F1)
    if(len(st)==len(population)):
        Pt1=st
    else:
        for l in len(F1)-1:
            F1=F1.union(F[l-1])
    
    
    

