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
    # dataset = 'Amazon670K'

    pg_folder = 'Output/Results/PGHash/'
    slide_folder = 'Output/Results/Slide/'
    dense_folder = 'Output/Results/Dense/'

    sw_labels = ['PGHash: 0.1CR', 'PGHash: 0.25CR', 'PGHash: 0.5CR', 'Full PGHash']
    sw_labels_dense = ['Dense Baseline', 'Dense Baseline', 'Dense Baseline', 'Dense Baseline']
    sw_crs = [0.1, 0.25, 0.5, 1.0]
    sw_crs_dense = [1.0, 1.0, 1.0, 1.0]
    # sw_tables = [200, 200, 200, 200]
    sw_tables = [50, 50, 50, 50]

    mw_crs = [1.0, 1.0, 1.0]
    mw_workers = [1, 4, 8]
    mw_tables_pg = [50, 50, 50]
    mw_tables_slide = [50, 50, 50]
    mw_rehash = 1
    mw_labels_pg = ['Single Device PGHash', '4 Device PGHash', '8 Device PGHash']
    mw_labels_slide = ['Single Device SLIDE', '4 Device SLIDE', '8 Device SLIDE']
    mw_labels_dense = ['Single Device FedAvg', '4 Device FedAvg', '8 Device FedAvg']

    mt_crs = [0.1, 0.25, 0.5, 1.0]
    mt_workers = [1]
    mt_tables = [5, 10, 50]
    mt_labels = ['PGHash: 5 Tables, ', 'PGHash: 10 Tables, ', 'PGHash: 50 Tables, ']
    mt_colors = ['r', 'g', 'b']

    amz_workers = [1, 4]

    multi_worker_test = True
    multi_cr = False
    multi_table = False
    avg_neuron = False

    if dataset == 'Delicious200K':
        # Delicious Results
        for trial in range(1, ntest+1):

            if multi_worker_test:
                for j in range(len(mw_workers)):

                    plt.figure()
                    cr = mw_crs[j]
                    pg_tables = mw_tables_pg[j]
                    slide_tables = mw_tables_slide[j]
                    nw = mw_workers[j]
                    pg_file = 'pg-pghash-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr-' + str(pg_tables) \
                              + 'tables-' + str(mw_rehash) + 'rehash'
                    slide_file = 'slide-slide-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr-' \
                                 + str(slide_tables) + 'tables-' + str(mw_rehash) + 'rehash'
                    dense_file = 'test1-regular-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr'

                    test_acc_pg, iters_pg = unpack_raw_test(pg_folder + pg_file)
                    test_acc_slide, iters_slide = unpack_raw_test(slide_folder + slide_file)
                    test_acc_dense, iters_dense = unpack_raw_test(dense_folder + dense_file)

                    plt.plot(iters_pg, test_acc_pg, label=mw_labels_pg[j], color='r')
                    plt.plot(iters_slide, test_acc_slide, label=mw_labels_slide[j], color='b')
                    plt.plot(iters_dense, test_acc_dense, label=mw_labels_dense[j], color='g')

                    plt.legend(loc='lower right')
                    plt.ylabel('Test Accuracy', fontsize=15)
                    plt.xlabel('Iterations', fontsize=15)
                    plt.xscale("log")
                    plt.xlim([1e2, 4e3])
                    plt.ylim([0.25, 0.48])
                    plt.grid(which="both", alpha=0.25)
                    # plt.show()
                    savefilename = 'multiworker' + str(nw) + '-' + str(mw_rehash) + 'rehash' + '.pdf'
                    plt.savefig(savefilename, format="pdf")
            elif multi_cr:
                for j in range(len(sw_crs)):

                    plt.figure()
                    tables = sw_tables[j]
                    cr = sw_crs[j]
                    cr2 = sw_crs_dense[j]
                    file = 'pg-pghash-' + dataset + '-' + '1workers-' + str(cr) + 'cr-' + str(tables) + 'tables'
                    # file = 'pg-pghash-' + dataset + '-' + '1workers-' + str(cr) + 'cr-' + str(tables) + 'tables-50rehash'
                    dense_file = 'test1-regular-' + dataset + '-1workers-' + str(cr2) + 'cr'
                    test_acc, iters = unpack_raw_test(pg_folder + file)
                    test_acc_d, iters_d = unpack_raw_test(dense_folder + dense_file)
                    plt.plot(iters, test_acc, label=sw_labels[j], color='r')
                    plt.plot(iters_d, test_acc_d, label=sw_labels_dense[j], color='g')
                    plt.legend(loc='lower right')
                    plt.ylabel('Test Accuracy', fontsize=15)
                    plt.xlabel('Iterations', fontsize=15)
                    plt.xscale("log")
                    plt.xlim([1e2, 4e3])
                    plt.ylim([0.25, 0.48])
                    plt.grid(which="both", alpha=0.25)
                    # plt.show()
                    savefilename = 'pg-varycr' + str(cr) + '.pdf'
                    plt.savefig(savefilename, format="pdf")

            elif multi_table:
                for j in range(len(mt_crs)):
                    cr = mt_crs[j]
                    plt.figure()
                    for k in range(len(mt_tables)):
                        tables = mt_tables[k]
                        color = mt_colors[k]
                        label = mt_labels[k] + str(cr) + 'CR'
                        file = 'pg-pghash-' + dataset + '-' + '1workers-' + str(cr) + 'cr-' + str(tables) + 'tables'
                        test_acc, iters = unpack_raw_test(pg_folder + file)
                        plt.plot(iters, test_acc, label=label, color=str(color))
                    plt.legend(loc='upper left')
                    plt.ylabel('Test Accuracy', fontsize=15)
                    plt.xlabel('Iterations', fontsize=15)
                    plt.xscale("log")
                    plt.xlim([1e2, 5e3])
                    plt.ylim([0.2, 0.48])
                    plt.grid(which="both", alpha=0.25)
                    # plt.show()
                    savefilename = 'pg-vary-tables' + str(cr) + '.pdf'
                    plt.savefig(savefilename, format="pdf")

            elif avg_neuron:

                print('hi')

    elif dataset == 'Amazon670K':

        for workers in amz_workers:

            plt.figure()
            pg_file = 'pg-pghash-' + dataset + '-' + str(workers) + 'workers-1.0cr-50tables-50rehash'
            slide_file = 'slide-slide-' + dataset + '-' + str(workers) + 'workers-1.0cr-50tables-50rehash'

            test_acc_pg, iters_pg = unpack_raw_test(pg_folder + pg_file)
            test_acc_slide, iters_slide = unpack_raw_test(slide_folder + slide_file)

            if workers == 1:
                legend_slide = 'Single Device SLIDE'
                legend_pg = 'Single Device PGHash'
            else:
                legend_slide = str(workers) + ' Device SLIDE'
                legend_pg = str(workers) + ' Device PGHash'

            plt.plot(iters_pg, test_acc_pg, label=legend_pg, color='r')
            plt.plot(iters_slide, test_acc_slide, label=legend_slide, color='b')

            plt.legend(loc='lower right')
            plt.ylabel('Test Accuracy', fontsize=15)
            plt.xlabel('Iterations', fontsize=15)
            plt.xscale("log")
            plt.grid(which="both", alpha=0.25)
            plt.xlim([100, 1.55e4])
            plt.ylim([0, 0.35])
            # plt.show()
            savefilename = 'amazon' + str(workers) + '-comparison-c8.pdf'
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


