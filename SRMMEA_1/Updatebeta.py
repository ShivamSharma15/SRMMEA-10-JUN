import random
import numpy as np
beta=[]
def update(NR,cluster):
    NRmax=max(NR)
    NRmin=min(NR)
    for i in range(0,cluster):
        if(NRmax==0):
            beta[i+1]=0
        else:
            beta[i+1]=NR[i]-NRmin/NRmax-NRmin
    
    return beta
