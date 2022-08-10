
import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
from numpy import sum

from matplotlib import cm
import statistics as s
import pandas as pd
from scipy import special



def B(t, T, gamma):

    Tau = T - t
    return (1-math.exp(-gamma * Tau))/gamma

def A(t, T, gamma, etha, sigma):

    Tau = T - t
    return (B(t, T, gamma) - Tau) * (gamma * etha - 0.5 * (sigma**2))/(gamma**2) - ((sigma * B(t, T, gamma))**2)/(4 * gamma)


def Y(r, t, T, gamma, etha, sigma):

    Tau = T - t
    if (T - t) == 0:
        return 1
    else:
        return - (A(t, T, gamma, etha, sigma) - r * B(t, T, gamma)) / Tau

def Yield():

    gamma = 0.25
    etha = 0.25 * 0.03
    sigma = 0.02
    r1 = 0.01
    r2 = 0.027
    r3 = 0.05

    T = np.zeros(31)
    Yield1 = np.zeros(31)
    Yield2 = np.zeros(31)
    Yield3 = np.zeros(31)

    for i in range(31):
        T[i] = i + 1
        Yield1[i] = Y(r1, 0, T[i], gamma, etha, sigma)
        Yield2[i] = Y(r2, 0, T[i], gamma, etha, sigma)
        Yield3[i] = Y(r3, 0, T[i], gamma, etha, sigma)

    #plt.plot(T, Yield2, label="Yield en fonction de T pour r = 0.027")
    #plt.plot(T, Yield1, label="Yield en fonction de T pour r = 0.01")
    plt.plot(T, Yield3, label="Yield en fonction de T pour r = 0.05")
    plt.legend()
    plt.show()

    print("Yield[1] = ", Yield1[0])
    print("Yield1[30]", Yield1[30])
    L = etha/gamma - 0.5 * ((sigma / gamma) ** 2)
    print("L est égale:", L)
    if abs(Yield1[1] - r1) < 0.01:
        print("Yield converge vers r lorsque T tend 0")
    if Yield1[30] - L < 0.01:
        print("Yield converge vers L lorsque T tend vers + infini")


def derivee_B(t, T, gamma):
    Tau = T - t
    return (Tau * math.exp(-gamma * Tau) - B(t, T, gamma)) / gamma

def derivee_A(t,T,gamma, etha, sigma):

    Tau = T - t

    return (1 / (gamma**2)) * (etha * (derivee_B(t, T, gamma) * gamma - B(t, T, gamma) ) + Tau * etha - (sigma**2) * (derivee_B(t, T, gamma) - 2 * B(t, T, gamma) / gamma)/2 - (sigma**2) * Tau / gamma - (sigma**2) * (B(t, T, gamma)/ 4) * (2 * gamma * derivee_B(t, T, gamma) - B(t, T, gamma) ) )


def derivee_etha(t, T, gamma):
    Tau = T - t
    return (B(t, T, gamma) - Tau) / (Tau * gamma)


def derivee_sigma_carre(t, T,gamma):

    Tau = T - t
    return - ((B(t, T, gamma) - Tau) / (2 * gamma) + (B(t, T, gamma) / 2)**2) / (Tau * gamma)
    #- (math.exp(- 2 * gamma * Tau) - 4 * math.exp(- gamma * Tau) - 2 * gamma * Tau + 3 ) / (4 * (gamma ** 3))


def derivee_gamma(r, t, T, gamma, etha, sigma):

    Tau = T - t
    return (derivee_A(t, T, gamma, etha, sigma) - r * derivee_B(t, T, gamma)) / Tau





def Vasicek():

    r = 0.04
    epsilon = 0.000000001
    Lamda = 0.01



    Y_Mar = [0.035, 0.041, 0.0439, 0.046, 0.0484, 0.0494, 0.0507, 0.0514, 0.052, 0.0523] ## Vecteur de taille 10
    Y_Mar1 = [0.056, 0.064, 0.074, 0.081, 0.082, 0.09, 0.087, 0.092, 0.0895, 0.091]
    T = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]) ## Vecteur de taille 10
    Y_th =[]
    Res = []
    J = np.zeros((10, 3))



    Identite = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    D = np.array([[1, 1, 1]]).T
    etha = 1
    gamma = 1
    sigma_carre = 1

    k = 0
    while (np.linalg.norm(D) > epsilon) :

        #print(np.linalg.norm(D))
        
            
        for p in range(10):

            Y_th.append(Y(r, 1, T[p], gamma, etha, cmath.sqrt(sigma_carre)))
            Res.append(Y_Mar1[p] - Y_th[p])
            Res_Matrix = np.array([Res]).T ## changer en matrice
            J[p][0] = derivee_etha(1, T[p], gamma)
            J[p][1] = derivee_sigma_carre(1, T[p], gamma)
            J[p][2] = derivee_gamma(r, 1, T[p], gamma, etha, cmath.sqrt(sigma_carre))

        D = - (np.linalg.inv(J.T@ J+ Lamda * Identite))@ J.T @ Res_Matrix  ## Produit matriciel

        J = np.zeros((10, 3))
        Y_th.clear()
        Res.clear()

        etha += D[0][0]
        sigma_carre += D[1][0]
        gamma += D[2][0]
        
        k = k + 1
    #Beta = [etha, sigma_carre, gamma]
    print(k)
    print("etha",etha,)
    print("sigma", cmath.sqrt(sigma_carre))
    print("gamma",gamma)


    ### Comparer Y_th et Y_Mar
    for i in range(10):
        Y_th.append(Y(r, 1, T[i], gamma, etha, cmath.sqrt(sigma_carre)))
    plt.plot(T, Y_th, label="Yth" )
    plt.plot(T, Y_Mar1, '.', label="Y_Mar1")
    plt.legend()
    plt.title("Comparaison Y_th et Y_Mar1 pout t=1")
    plt.show()


def calibration_hist_dates():

    T = 5
    etha = 0.6
    gamma = 4
    sigma = 0.08
    N = 100
    delta_t = T/N
    X = []
    X_carre = []
    Y = []
    Y_th = []
    Z = []
    r = np.zeros(100)
    t = np.linspace(0, 1, 100)
    r[0] = 0.05

    for i in range(99):

        X.append(r[i])
        X_carre.append(r[i] ** 2)
        Y_th.append(X[i] * 0.8817867419193822 + 0.017808406981631)
        r[i+1] = r[i] * math.exp(-gamma * delta_t) + etha * (1 - math.exp(-gamma * delta_t))/gamma + sigma * math.sqrt((1 - math.exp(-2 * gamma * delta_t))/(2 * gamma)) * np.random.normal(0, 1)
        Y.append(r[i + 1])
        Z.append(r[i] * r[i + 1])



    plt.plot(t, r)
    plt.xlabel("t")
    plt.ylabel("r")
    plt.title("Taux d'intéret en fct du temps")
    plt.show()

    plt.plot(X, Y, '.')
    plt.plot(X, Y_th, label="Droite de régression linéaire")
    plt.legend()
    plt.title("Nuages de points de Y en Fct de X")
    plt.show()
###
    beta1 = (sum(Z) - 0.01 * sum(Y) * sum(X)) / (sum(X_carre) - 0.01 * (sum(X) ** 2))
    beta2 = (sum(Y) * sum(X_carre) - sum(X) * sum(Z)) / (100 * sum(X_carre) - sum(X) ** 2)
    print("La valeur de Beta1 est:", beta1)
    print("La valeur de Beta2 est:", beta2)

    D_carre = 0
    for i in range(99):
        D_carre += 0.01 * (Y[i] - (beta1 * X[i] + beta2))**2
    print("D au carré est:",D_carre)

    gamma = - math.log(beta1) / delta_t
    etha = gamma * beta2 / (1 - beta1)
    sigma = cmath.sqrt(-D_carre * 2 * math.log(beta1) / (delta_t * 1 - beta1**2))
    print("etha est : ",etha)
    print("gamma est:", gamma)
    print("sigma est :",sigma)

if __name__ == '__main__':
    Yield()
    Vasicek()
    calibration_hist_dates()
