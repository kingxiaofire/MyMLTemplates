import numpy as np

def markov():
    init_arrary = np.array([0.9, 0.02, 0.08])
    transfer_matrix = np.array([[0.9, 0.075, 0.025],
                                [0.15, 0.8, 0.05],
                                [0.25, 0.25, 0.5]])
    restemp = init_arrary
    for i in range(220):
        res = np.dot(restemp, transfer_matrix)
        print(i,"\t", res)
        restemp = res

markov()