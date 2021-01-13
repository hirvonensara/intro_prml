# Introduction to Pattern Recognition and Machine Learning
# Exercise 1 - Fitting a line
# Sara Hirvonen

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def my_linfit(x,y):
    N = len(x)
    a = 0
    b = 0
    
    # solve sigma(y*x)
    sigma_yx = 0
    for i in range(0, N):
        sigma_yx = sigma_yx + y[i]*x[i]
        
    # solve sigma(x^2)
    sigma_x2 = 0
    for j in range(0, N):
        sigma_x2 = sigma_x2 + pow(x[j], 2)
    
    a = (N*sigma_yx-sum(y)*sum(x))/(N*sigma_x2-pow(sum(x),2))
    b = (sum(y)*sigma_x2-sigma_yx*sum(x))/(N*sigma_x2-pow(sum(x),2))

    return a, b

def main():
    x = []
    y = []

    #plt.figure(1)
    plt.ylim(-5, 5)
    plt.xlim(-2, 5)
    plt.setp(plt.gca(), autoscale_on=False)
    plt.figtext(0.2, 0.9,"Select points by left click and stop collecting by right click")
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_pop=None, mouse_stop=3)
   
    for p in pts:
        x.append(p[0])
        y.append(p[1])
 
    a,b = my_linfit(x,y)

    plt.plot(x,y, 'kx')
    xp = np.arange(-2,5,0.1)
    plt.plot(xp, a*xp+b)
    plt.figtext(0.1,0,"My fit:a = {} and b = {}".format(a, b))
    plt.show()

main()