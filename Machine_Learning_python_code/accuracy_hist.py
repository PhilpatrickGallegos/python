import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()


def main():
    linReg = np.loadtxt('Hep_Ln_GD.dat')
    non_linReg = np.loadtxt('Hep_Log_Reg.dat')
    logReg = np.loadtxt('Nl_Reg_Paraboloid.dat')
    descTree = np.loadtxt('Decision_Tree.dat')

    accur = np.hstack([linReg,non_linReg,logReg,descTree])
    ml_algorithms = ('Linear Regression','Non-Linear Regression',
                     'Logistic Regression', 'Descision Tree')
    y_pos = np.arange(len(ml_algorithms))

    plt.bar(y_pos,accur,align = 'center', alpha = 0.5)
    plt.xticks(y_pos,ml_algorithms)
    plt.ylabel("Accuracy")
    plt.title("Hepatitis Data")
    plt.show()



main()
