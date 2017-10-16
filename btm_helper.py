import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, t


def s_xx(x):
    return np.sum((x - np.mean(x))**2)


def s_xy(x, y):
    return np.sum(y * (x - np.mean(x)))


def reg(x, y):
    beta_1 = s_xy(x, y) / s_xx(x)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return np.array([beta_0, beta_1])


def pred(x, beta):
    X = np.vstack([np.ones_like(x), x]).T
    return np.dot(X, beta)


def mse(x, y):
    beta = reg(x, y)
    yhat = pred(x, beta)
    ss = np.sum((y - yhat)**2)
    return np.sqrt(1 / (x.shape[0] - 2) * ss)


def beta1_t_stat(x, y):
    beta_0, beta_1 = reg(x, y)
    return (beta_1 * np.sqrt(s_xx(x))) / mse(x, y)


def test_data_array(data_array):
    data = np.loadtxt('sales_data.txt')
    assert(np.array_equal(data_array, data)), 'There seems to be something\
    wrong with your data import.'
    return 'Good job!'


def test_x_and_y(x, y):
    data = np.loadtxt('sales_data.txt')
    assert(np.array_equal(x, data[:, 1])), 'There is something wrong with\
    x. Please check it again!'
    assert(np.array_equal(y, data[:, 0])), 'There is something wrong with\
    y, Please check it again!'
    return 'Both x and y look good! Good job!'


def test_simple_linear_regression(beta_0, beta_1):
    assert(np.isclose(beta_1, 2.77211593))
    assert(np.isclose(beta_0, -157.33011359))
    return 'Perfect! Your calculations are right!'


def test_predict(func):
    data = np.loadtxt('sales_data.txt')
    beta = reg(data[:, 0], data[:, 1])
    assert(np.isclose(func(400, beta[0], beta[1]), pred(400, beta))), 'Seriously ?!'
    return 'Looks Good :)!'


def test_t_stat(beta_stat):
    data = np.loadtxt('sales_data.txt')
    stat = beta1_t_stat(data[:, 0], data[:, 1])
    assert(np.isclose(beta_stat, stat)), 'Nooo, try harder!'
    return 'Like a boss! Awesome!'


def test_p_val(p_val):
    data = np.loadtxt('sales_data.txt')
    stat = beta1_t_stat(data[:, 0], data[:, 1])
    pval = 1 - t.cdf(stat, df=data.shape[0] - 2)
    assert(np.isclose(p_val, pval)), 'Nooo, try harder!'
    return 'Excellent!'


def investigate_mse(size, std, beta_0=-157, beta_1=3, nbounds=500):
    # this is the real function
    def f(x): return beta_0 + beta_1 * x

    # we simulate the files
    x = uniform.rvs(loc=374, scale=300, size=size)
    y = norm.rvs(loc=f(x), scale=std)

    # estimate the perform regression on simulated data
    hat_beta_0, hat_beta_1 = reg(x, y)

    def hat_f(x): return hat_beta_0 + hat_beta_1 * x

    # xs to plot the regression lines
    z = np.linspace(350, 700, 1000)

    # calculate mse
    hat_std = mse(x, y)
    g = np.linspace(-nbounds, nbounds, 1000)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(x, y, 'o')
    ax[0].plot(z, f(z), label='$f(x)=${b0}+{b1}$x$'.format(b0=beta_0, b1=beta_1))
    ax[0].plot(z, hat_f(z), '--',
               label='$\hat f(x)=${b0}+{b1}$x$'.format(b0=int(hat_beta_0), b1=int(hat_beta_1)))

    for i, x_i in enumerate(x):
        ax[0].vlines(x=x_i, ymin=y[i], ymax=hat_f(x_i), color='red', linewidth=1)

    ax[0].set_xlabel('Advertisment')
    ax[0].set_ylabel('Sales')
    #ax.set_ylim([700, 2000])
    ax[0].legend()

    ax[1].plot(g, norm.pdf(g, loc=0, scale=hat_std), color='red',
               linewidth=2, label='Estimated MSE={}'.format(int(hat_std)))
    for i, x_i in enumerate(y - hat_f(x)):
        ax[1].vlines(x=x_i, ymin=0, ymax=ax[1].get_ylim()[1] * 0.05, color='red', linewidth=1)
    ax[1].set_xlabel('$\epsilon_i$')
    ax[1].set_ylabel('Probability of $\epsilon_i$ ($\epsilon_i\sim\mathcal{N}(0, MSE)$)')
    ax[1].legend()
