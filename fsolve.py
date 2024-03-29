from scipy.optimize import fsolve  
from math import cos
import pysam

def f(x):  
    d = 140  
    l = 156  
    a = float(x[0])  
    r = float(x[1])  
    return [  
        cos(a) - 1 + (d*d)/(2*r*r), 
        l - r * a                   
    ]  
result = fsolve(f, [1, 1])  
print (result ) 