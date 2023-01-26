import matplotlib.pyplot as plt
import numpy as np
import os
import tikzplotlib


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 1000, low_percentile = 1, high_percentile = 99):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(x, y, num_runs, num_dots, mylegend,ls='-', lw=3, transparency=0.2):
    assert(x.ndim==1)
    assert(x.size==num_dots)
    assert(y.ndim==2)
    assert(y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    plt.plot(x, y_mean, 'o-', label=mylegend, linestyle=ls, linewidth=lw) #, label=r'$\alpha$={}'.format(alpha))
    plt.fill_between(x, y_min, y_max, alpha=transparency)
    return


def unpack_data(directory_path, datatype='losses.log', epochs=200, num_workers=10):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    data = np.zeros((epochs, num_workers))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                j = int(file.split('-')[0][1:])
                with open(directory_path + '/' + file, 'r') as f:
                    i = 0
                    for line in f:
                        data[i, j] = line
                        i += 1
    return data


def unpack_raw_test(directory_path, file_test='r0-test-acc-top1.log'):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    test_acc_raw = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_test):
                with open(directory_path + '/' + file, 'r') as f:
                    for line in f:
                        acc = line.split('\n')[0]
                        test_acc_raw.append(float(acc))

    acc_iter = []
    test_acc = []
    for i, val in enumerate(test_acc_raw, 1):
        if not np.isnan(val):
            if not(val in test_acc):
                acc_iter.append(i)
                test_acc.append(val)

    return np.array(test_acc), np.array(acc_iter)



if __name__ == '__main__':
    colors = ['r', 'b', 'g', 'orange', 'pink', 'cyan', 'yellow', 'purple']
    ntest = 1

    # specify number of epochs
    epochs = 200
    # specify number of epochs to plot
    plot_epochs = 200
    # specify number of workers
    num_workers = 1

    # specify which statistics to graph on the y-axis (will make separate plots)
    stats = 'test-acc-top1.log'
    dataset = 'Delicious200K'

    pgfolder = 'Output/PGResults/'
    fedavg_folder = 'Output/TrueResults/'
    folder = 'Output/ICML-Results/'

    sw_labels = ['PGHash: 0.1CR', 'PGHash: 0.25CR', 'PGHash: 0.5CR']
    sw_crs = [0.1, 0.25, 0.5]

    mw_crs = [0.1, 0.1, 0.1]
    mw_workers = [1, 4, 8]
    mw_labels = ['PGHash: 0.1CR, 1 Worker', 'PGHash: 0.1CR, 4 Workers', 'PGHash: 0.1CR, 8 Workers']

    multi_worker_test = False

    # Delicious Results

    for trial in range(1, ntest+1):

        if multi_worker_test:
            for j in range(len(mw_workers)):

                plt.figure()
                cr = 0.1
                nw = mw_workers[j]
                file = 'pghash-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr'

                test_acc_pg, iters_pg = unpack_raw_test(folder+file)

                plt.plot(iters_pg, test_acc_pg, label=mw_labels[j], color='r')

                # plot dense baseline
                dense_file = 'dense-' + dataset + '-' + str(nw) + 'workers-' + '1.0cr'
                baseline_filepath = folder + dense_file
                test_acc, iters = unpack_raw_test(baseline_filepath)
                if nw == 1:
                    leg = 'Dense Baseline: 1 Worker'
                else:
                    leg = 'FedAvg: ' + str(nw) + ' Workers'

                if j == 2:
                    iters = iters[:len(iters_pg)]
                    test_acc = test_acc[:len(test_acc_pg)]
                plt.plot(iters, test_acc, label=leg, color='b')
                plt.legend(loc='best')
                plt.ylabel('Test Accuracy', fontsize=15)
                plt.xlabel('Iterations', fontsize=15)
                plt.xscale("log")
                if j == 0:
                    plt.xlim([1e2, 8e3])
                else:
                    plt.xlim([1e2, 6.1e3])
                plt.ylim([0.10, 0.48])
                plt.grid(which="both", alpha=0.25)
                # plt.show()
                savefilename = 'pg-multiworker' + str(nw) + '.pdf'
                plt.savefig(savefilename, format="pdf")
        else:
            for j in range(len(sw_crs)):

                plt.figure()
                cr = sw_crs[j]
                file = 'pghash-' + dataset + '-' + '1workers-' + str(cr) + 'cr'
                test_acc, iters = unpack_raw_test(folder + file)
                plt.plot(iters, test_acc, label=sw_labels[j], color='r')
                plt.legend(loc='best')
                plt.ylabel('Test Accuracy', fontsize=15)
                plt.xlabel('Iterations', fontsize=15)
                plt.xscale("log")
                plt.xlim([1e2, 8e3])
                plt.ylim([0.25, 0.48])
                plt.grid(which="both", alpha=0.25)
                # plt.show()
                savefilename = 'pg-varycr' + str(cr) + '.pdf'
                plt.savefig(savefilename, format="pdf")


    '''
    for trial in range(1, ntest+1):

        if single_worker_test:
            for j in range(len(sw_tables)):
                cr = sw_crs[j]
                if sw_tables[j] == 1:
                    file = 'test' + str(trial) + '-pghash-' + dataset + '-' + str(num_workers) + 'workers-' \
                           + str(cr) + 'cr'
                else:
                    file = 'test' + str(trial) + '-pghash-' + dataset + '-' + str(num_workers) + 'workers-' + str(cr) \
                           + 'cr-' + str(sw_tables[j]) + 't'
                test_acc, iters = unpack_raw_test(pgfolder+file)

                plt.plot(iters, test_acc, label=sw_labels[idx], color=colors[idx])
                idx += 1

            # plot dense baseline
            baseline_filepath = fedavg_folder + 'test1-regular-Delicious200K-1workers-1.0cr'
            test_acc, iters = unpack_raw_test(baseline_filepath)
            plt.plot(iters, test_acc, label='Dense Baseline', color=colors[idx])
            plt.legend(loc='best')
            plt.ylabel('Test Accuracy', fontsize=14)
            plt.xlabel('Iterations', fontsize=14)
            plt.xscale("log")
            plt.xlim([1e2, 1e4])
            plt.grid()
            # plt.show()
            plt.savefig("pg1worker.pdf", format="pdf")
        else:
            for j in range(len(mw_labels)):
                cr = mw_crs[j]
                nw = mw_workers[j]
                file = 'test' + str(trial) + '-pghash-' + dataset + '-' + str(nw) + 'workers-' \
                       + str(cr) + 'cr'
                test_acc, iters = unpack_raw_test(pgfolder + file)
                plt.plot(iters, test_acc, label=mw_labels[idx], color=colors[idx])
                idx += 1
            # plot dense baseline
            for j in range(len(mw_labels)):
                nw = mw_workers[j]
                baseline_filepath = fedavg_folder + 'test1-regular-Delicious200K-' + str(nw) + 'workers-1.0cr'
                test_acc, iters = unpack_raw_test(baseline_filepath)
                if j == 0:
                    leg = 'Dense Baseline: 1 Worker'
                else:
                    leg = 'FedAvg: ' + str(nw) + ' Workers'
                plt.plot(iters, test_acc, label=leg, color=colors[idx])
                idx += 1
            plt.legend(loc='best')
            plt.ylabel('Test Accuracy', fontsize=14)
            plt.xlabel('Iterations', fontsize=14)
            plt.xscale("log")
            plt.xlim([1e2, 1e4])
            plt.grid()
            # plt.show()
            plt.savefig("pg-multiworker.pdf", format="pdf")
    '''


