'''
Created on 27. okt. 2017

@author: ljb
'''
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from timeit import default_timer as timer

mpl.rcParams['legend.fontsize'] = 10


def plot_earth_orbit3d(ax, x, y, z, label='earth orbit'):
    ax.plot(x, y, label=label)
    ax.axis('equal')
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_zlabel('z [AU]')


def plot_earth_orbit(x, y, label='earth orbit'):
    plt.plot(x, y, label=label)
    plt.axis('equal')
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.legend(loc='best')
    #plt.plot(t, dx, t, dy, label='')
    plt.legend(loc='best')


def solve_timer(problem, solverclass, T=100):
    dt = problem.dt
    t = np.arange(0, T, dt)
    # solver = ForwardEuler(problem)
    solver = solverclass(problem)
    v0 = getattr(problem, 'v0', None)
    if v0 is None:
        solver.set_initial_condition(problem.U0)
    else:
        solver.set_initial_condition(problem.U0, v0=v0)
    t0 = timer()
    U, t = solver.solve(t)
    elapsed_time = timer() - t0
    return U, t, elapsed_time, solver


def solve_and_plot_earth_and_jupiter_orbit(problem, solverclass, T=100):
    U, t, elapsed_time, solver = solve_timer(problem, solverclass, T)
    classname = solver.name
    beta = np.round(100 * problem.beta)
    k = int(np.round(np.abs(np.log2(problem.dt))))

    x, y, z, xj, yj, zj = U[:, :6].T
    k_txt = '{' + '{}'.format(k) + '}'
    msg = 'dt = $1/2^{}$[year], M_sun/M_jupiter = {:3.3g}'.format(k_txt, 1. / problem.gm[1])
    if True:
        # fig = plt.figure()
        filename = 'stability_2d_test_k{}beta{}{}.png'.format(k, beta, classname)
        # plt.hold('on')
        plot_earth_orbit(x, y, label='Earth {}'.format(msg))
        plot_earth_orbit(xj, yj, label='Jupiter {}'.format(msg))
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_earth_orbit3d(ax, x, y, z, label='Earth')
        plot_earth_orbit3d(ax, xj, yj, zj, label='Jupiter')
        ax.legend(loc='best')
        filename = 'stability_3d_test_k{}beta{}{}.png'.format(k, beta, classname)

    #plt.suptitle('{} method'.format(classname))
    # plt.suptitle(msg)
    print(filename)
    plt.savefig(filename)

    results = dict(k=k, elapsed_time=elapsed_time, method=classname, beta=problem.beta)
    return results


def solve_and_plot_earth_jupiter_and_sun_orbit(problem, solverclass, T=100):
    U, t, elapsed_time, solver = solve_timer(problem, solverclass, T)
    classname = solver.name
    beta = np.round(100 * problem.beta)
    k = int(np.round(np.abs(np.log2(problem.dt))))


    x, y, z, xj, yj, zj = U[:, :6].T
    xs, ys, zs = U[:, -3:].T
    k_txt = '{' + '{}'.format(k) + '}'
    msg = 'dt = $1/2^{}$[year], M_sun/M_jupiter = {:3.3g}'.format(k_txt, 1. / problem.gm[1])
    msg = ''

    if True:
        fig = plt.figure()
        filename = 'stability_2d_test_k{}beta{}{}.png'.format(k, beta, classname)
        # plt.hold('on')
        plot_earth_orbit(x, y, label='Earth {}'.format(msg))
        plot_earth_orbit(xj, yj, label='Jupiter {}'.format(msg))
        plot_earth_orbit(xs, ys, label='Sun {}'.format(msg))
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_earth_orbit3d(ax, x, y, z, label='Earth')
        plot_earth_orbit3d(ax, xj, yj, zj, label='Jupiter')
        ax.legend(loc='best')
        filename = 'stability_3d_test_k{}beta{}{}.png'.format(k, beta, classname)

    #plt.suptitle('{} method'.format(classname))
    # plt.suptitle(msg)
    print(filename)
    plt.savefig(filename)

    results = dict(k=k, elapsed_time=elapsed_time, method=classname, beta=problem.beta)
    return results


def solve_and_plot_all_planet_orbits(problem, solverclass, T=100, names=None):
    U, t, elapsed_time, solver = solve_timer(problem, solverclass, T)
    classname = solver.name
    beta = np.round(100 * problem.beta)
    k = int(np.round(np.abs(np.log2(problem.dt))))

    k_txt = '{' + '{}'.format(k) + '}'
    msg = 'dt = $1/2^{}$[year], M_sun/M_jupiter = {:3.3g}'.format(k_txt, 1. / problem.gm[1])
    msg = ''

    if False:
        fig = plt.figure()
        filename = 'stability_2d_test_k{}beta{}{}.png'.format(k, beta, classname)
        for i, name in enumerate(names):
            x, y, z = U[:, 3*i:3*(i+1)].T
            plot_earth_orbit(x, y, z, label=name)
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, name in enumerate(names):
            x, y, z = U[:, 3*i:3*(i+1)].T
            plot_earth_orbit3d(ax, x, y, z, label=name)
        ax.legend(loc='best')
        filename = 'stability_3d_test_k{}beta{}{}.png'.format(k, beta, classname)

    #plt.suptitle('{} method'.format(classname))
    # plt.suptitle(msg)
    print(filename)
    plt.savefig(filename)

    results = dict(k=k, elapsed_time=elapsed_time, method=classname, beta=problem.beta)
    return results



def solve_and_animate_earth_and_jupiter_orbit(problem, solverclass, T=100):
    U, t, elapsed_time, solver = solve_timer(problem, solverclass, T)
    classname = solver.name
    mass_sun_jupiter_ratio = (1. / problem.gm[1])
    k = int(np.round(np.abs(np.log2(problem.dt))))

    x, y, z, xj, yj, zj = U[:, :6].T
    k_txt = '{' + '{}'.format(k) + '}'
    msg = 'dt = $1/2^{}$[year], M_sun/M_jupiter = {:3.3g}'.format(k_txt, mass_sun_jupiter_ratio)

    # fig = plt.figure()
    ani = plot_animation(x, y, xj, yj, problem.dt, title=msg)

    filename = 'animate_stability_2d_test_Mj{}beta{}{}.mp4'.format(
        k, mass_sun_jupiter_ratio, classname)
    print(filename)
    # ani.save(filename, fps=15)
    plt.show('hold')
    #plt.suptitle('{} method'.format(classname))
    # plt.suptitle(msg)
    #     print(filename)
    #     plt.savefig(filename)

    results = dict(k=k, elapsed_time=elapsed_time, method=classname, beta=problem.beta)
    return results


def solve_and_plot_earth_orbit(problem, solverclass, T=100):
    U, t, elapsed_time, solver = solve_timer(problem, solverclass, T)
    classname = solver.name
    beta = np.round(100 * problem.beta)
    k = np.round(np.abs(np.log2(problem.dt)))
    x, y = U[:, :2].T
    plot_earth_orbit(x, y)
    filename = 'stability_test_k{}beta{}{}.png'.format(k, beta, classname)
    plt.suptitle('{} method used {} seconds'.format(classname, elapsed_time))
    plt.title('dt = {}[year], beta = {}'.format(problem.dt, problem.beta))
    print(filename)
    plt.savefig(filename)

    results = dict(k=k, elapsed_time=elapsed_time, method=classname, beta=problem.beta)
    return results


def plot_animation(x1, y1, x2, y2, dt, title='', xlim=None, ylim=None):
    fig = plt.figure()
    fact = 2
    if xlim is None:
        xlim = list((np.min(x2)*fact, fact*np.max(x2)))
    if ylim is None:
        ylim = list((np.min(y2)*fact, fact*np.max(y2)))
    xlim = list(xlim)
    ylim = list(ylim)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
    ax.grid()

    line1, = ax.plot([], [], 'yo', lw=2, label='sun')
    line2, = ax.plot([], [], 'bo', lw=2, label='Earth')
    line3, = ax.plot([], [], 'ro', lw=2, label='Jupiter')
    # ax.axis('equal')
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    if title:
        ax.set_title(title)
    ax.legend(loc='best')

    time_template = 'time = %.1f years'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, time_text

    def animate(i):

        line1.set_data([0], [0])
        line2.set_data([x1[i]], [y1[i]])
        line3.set_data([x2[i]], [y2[i]])

#         if x1[i] <xlim[0]:
#             xlim[0] = x1[i]
#
#         if xlim[1]<x1[i]:
#             xlim[1] = x1[i]
#
#         if y1[i] <ylim[0]:
#             ylim[0] = y1[i]
#
#         if ylim[1]<y1[i]:
#             ylim[1] = y1[i]
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)

        time_text.set_text(time_template % (i * dt))
        return line1, line2, line3, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(x1)),
                                  interval=1, blit=True, init_func=init)
    return ani
#


if __name__ == '__main__':
    pass
