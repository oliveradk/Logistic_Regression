#plots cost function from dataset of cost and # of iterations
import matplotlib.pyplot as plt
def plotCost(costData):
    cost = [x[0] for x in costData]
    iter = [x[1] for x in costData]
    'Plots cost function against iterations'
    plt.plot(cost, iter)
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()
