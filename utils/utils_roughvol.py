import warnings
warnings.filterwarnings('ignore')

import numpy as np
from fbm import FBM

def fOU_generator(a,n=0.3,h=0.2,length=300):
    
    fbm_increments = np.diff(FBM(length, h).fbm())
    # X(t+1) = X(t) - a(X(t)-m) + n(W(t+1)-W(t))
    x0 = np.random.normal(1,0.1)
    x0 = 0.5
    m = x0
    price = [x0]
    for i in range(length):
        p = price[i] - a*(price[i]-m) + n*fbm_increments[i]
        price.append(p)
    return np.array(price)
