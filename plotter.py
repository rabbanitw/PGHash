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
    # number of test runs
    ntest = 3
    # specify number of workers
    num_workers = 1

    # specify which statistics to graph on the y-axis (will make separate plots)
    stats = 'test-acc-top1.log'
    datasets = ['Delicious200K', 'Amazon670K']
    # dataset = 'Delicious200K'
    # dataset = 'Amazon670K'
    dataset = 'Wiki325K'

    pg_folder = 'output/Results/PGHash/'
    slide_folder = 'output/Results/Slide/'
    dense_folder = 'output/Results/Dense/'

    sw_labels = ['PGHash: 0.1CR', 'PGHash: 0.25CR', 'Full PGHash']
    sw_labels_dense = ['Dense Baseline', 'Dense Baseline', 'Dense Baseline', 'Dense Baseline']
    sw_crs = [0.1, 0.25, 1.0]
    sw_crs_dense = [1.0, 1.0, 1.0]
    # sw_tables = [200, 200, 200, 200]
    sw_tables = [50, 50, 50, 50]

    mw_crs = [1.0, 1.0, 1.0]
    mw_workers = [1, 4, 8]
    # mw_workers = [1, 8]

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
    sampled_softmax = False

    ds = dataset + '/'

    if dataset == 'Delicious200K':
        # Delicious Results

        if multi_worker_test:
            for j in range(len(mw_workers)):

                cr = mw_crs[j]
                pg_tables = mw_tables_pg[j]
                slide_tables = mw_tables_slide[j]
                nw = mw_workers[j]

                slide_accs = []
                dense_accs = []
                pg_accs = []
                dense_iters = []
                max_iters = 4.05e3
                cutoff = 0

                for trial in range(1, ntest + 1):

                    pg_file = 'run' + str(trial) + '-pghash-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr-' \
                              + str(pg_tables) + 'tables-' + str(mw_rehash) + 'rehash'
                    slide_file = 'run' + str(trial) + '-slide-' + dataset + '-' + str(nw) + 'workers-' + str(cr) \
                                 + 'cr-' + str(slide_tables) + 'tables-' + str(mw_rehash) + 'rehash'
                    dense_file = 'test' + str(trial) + '-regular-' + dataset + '-' + str(nw) + 'workers-' + str(cr) + 'cr'

                    test_acc_pg, iters_pg = unpack_raw_test(pg_folder + ds + pg_file)
                    test_acc_slide, iters_slide = unpack_raw_test(slide_folder + ds + slide_file)
                    test_acc_dense, iters_dense = unpack_raw_test(dense_folder + ds + dense_file)

                    cutoff = np.max((np.count_nonzero(iters_pg < max_iters),
                                    np.count_nonzero(iters_slide < max_iters),
                                    np.count_nonzero(iters_dense < max_iters), 0))

                    slide_accs.append(test_acc_slide)
                    dense_accs.append(test_acc_dense)
                    pg_accs.append(test_acc_pg)
                    dense_iters.append(iters_dense)

                for trial in range(0, ntest):
                    slide_accs[trial] = slide_accs[trial][:cutoff]
                    dense_accs[trial] = dense_accs[trial][:cutoff]
                    pg_accs[trial] = pg_accs[trial][:cutoff]
                    lp = len(pg_accs[trial])
                    if lp < cutoff:
                        pg_accs[trial] = np.append(pg_accs[trial], np.mean(np.array([pg_accs[trial-2][lp:cutoff],
                                                                                     pg_accs[trial-1][lp:cutoff]]),
                                                                           axis=0))
                iters = iters_pg[:cutoff]

                slide_accs = np.stack(slide_accs, axis=0)
                pg_accs = np.stack(pg_accs, axis=0)

                if nw == 8:
                    dense_accs[1] = np.append(dense_accs[1], (dense_accs[0][-1] + dense_accs[2][-1]) / 2)

                dense_accs = np.stack(dense_accs, axis=0)

                y_mean_s, y_min_s, y_max_s = generate_confidence_interval(slide_accs)
                y_mean_p, y_min_p, y_max_p = generate_confidence_interval(pg_accs)
                y_mean_d, y_min_d, y_max_d = generate_confidence_interval(dense_accs)

                plt.figure()

                plt.plot(iters, y_mean_p, label=mw_labels_pg[j], color='r')
                plt.fill_between(iters, y_min_p, y_max_p, alpha=0.2, color='r')
                plt.plot(iters, y_mean_s, label=mw_labels_slide[j], color='b')
                plt.fill_between(iters, y_min_s, y_max_s, alpha=0.2, color='b')

                if nw != 8:
                    plt.plot(iters, y_mean_d, label=mw_labels_dense[j], color='g')
                    plt.fill_between(iters, y_min_d, y_max_d, alpha=0.2, color='g')
                else:
                    plt.plot(dense_iters[0], y_mean_d, label=mw_labels_dense[j], color='g')
                    plt.fill_between(dense_iters[0], y_min_d, y_max_d, alpha=0.2, color='g')

                plt.legend(loc='lower right')
                plt.ylabel('Test Accuracy', fontsize=15)
                plt.xlabel('Iterations', fontsize=15)
                plt.xscale("log")
                plt.xlim([1e2, 4e3])
                plt.ylim([0.225, 0.48])
                plt.grid(which="both", alpha=0.25)
                # plt.show()
                savefilename = 'multiworker' + str(nw) + '-' + str(mw_rehash) + 'rehash' + '.pdf'
                plt.savefig(savefilename, format="pdf")
        elif multi_cr:
            for j in range(len(sw_crs)):

                tables = sw_tables[j]
                cr = sw_crs[j]
                cr2 = sw_crs_dense[j]
                dense_accs = []
                pg_accs = []
                max_iters = 4.05e3

                for trial in range(1, ntest + 1):

                    file = 'run' + str(trial) + '-pghash-' + dataset + '-' + '1workers-' + str(cr) + 'cr-' \
                           + str(tables) + 'tables-1rehash'
                    dense_file = 'test' + str(trial) + '-regular-' + dataset + '-1workers-' + str(cr2) + 'cr'
                    test_acc, iters = unpack_raw_test(pg_folder + ds + file)
                    test_acc_d, iters_d = unpack_raw_test(dense_folder + ds + dense_file)

                    cutoff = np.count_nonzero(iters < max_iters)
                    dense_accs.append(test_acc_d[:cutoff])
                    pg_accs.append(test_acc[:cutoff])

                iters = iters[:cutoff]
                pg_accs = np.stack(pg_accs, axis=0)
                dense_accs = np.stack(dense_accs, axis=0)
                y_mean_p, y_min_p, y_max_p = generate_confidence_interval(pg_accs)
                y_mean_d, y_min_d, y_max_d = generate_confidence_interval(dense_accs)

                plt.figure()
                plt.plot(iters, y_mean_p, label=sw_labels[j], color='r')
                plt.fill_between(iters, y_min_p, y_max_p, alpha=0.2, color='r')
                plt.plot(iters, y_mean_d, label=sw_labels_dense[j], color='g')
                plt.fill_between(iters, y_min_d, y_max_d, alpha=0.2, color='g')

                # plt.plot(iters, test_acc, label=sw_labels[j], color='r')
                # plt.plot(iters_d, test_acc_d, label=sw_labels_dense[j], color='g')

                plt.legend(loc='lower right')
                plt.ylabel('Test Accuracy', fontsize=15)
                plt.xlabel('Iterations', fontsize=15)
                plt.xscale("log")
                plt.xlim([1e2, 4e3])
                plt.ylim([0.225, 0.48])
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
                    test_acc, iters = unpack_raw_test(pg_folder + ds + file)
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

    elif dataset == 'Amazon670K':

        for workers in amz_workers:

            slide_accs = []
            slide_iters = []
            pg_accs = []
            pg_iters = []
            max_iter = int(1.5e4)
            count_s = 0
            count_p = 0

            for trial in range(1, ntest + 1):

                pg_file = 'run' + str(trial) + '-pghash-' + dataset + '-' + str(workers) + \
                          'workers-1.0cr-50tables-50rehash'
                slide_file = 'run' + str(trial) + '-slide-' + dataset + '-' + str(workers) + \
                             'workers-1.0cr-50tables-50rehash'

                test_acc_pg, iters_pg = unpack_raw_test(pg_folder + ds +pg_file)
                test_acc_slide, iters_slide = unpack_raw_test(slide_folder + ds + slide_file)

                slide_accs.append(test_acc_slide)
                pg_accs.append(test_acc_pg)
                pg_iters.append(iters_pg)
                slide_iters.append(iters_slide)

            maxx_p = max(x.shape[0] for x in pg_accs)
            maxx_s = max(x.shape[0] for x in slide_accs)
            iters_pg = pg_iters[np.argmax([x.shape[0] for x in pg_iters])]
            iters_slide = slide_iters[np.argmax([x.shape[0] for x in slide_iters])]

            for trial in range(0, ntest):
                pg_accs[trial].resize(maxx_p, refcheck=False)
                slide_accs[trial].resize(maxx_s, refcheck=False)

            slide_accs = np.stack(slide_accs, axis=0)
            pg_accs = np.stack(pg_accs, axis=0)
            pg_accs[pg_accs == 0] = np.NaN
            slide_accs[slide_accs == 0] = np.NaN

            col_mean_p = np.nanmean(pg_accs, axis=0)
            col_mean_s = np.nanmean(slide_accs, axis=0)

            # Find indices that you need to replace
            inds_p = np.where(np.isnan(pg_accs))
            inds_s = np.where(np.isnan(slide_accs))

            # Place column means in the indices. Align the arrays using take
            pg_accs[inds_p] = np.take(col_mean_p, inds_p[1])
            slide_accs[inds_s] = np.take(col_mean_s, inds_s[1])

            y_mean_s, y_min_s, y_max_s = generate_confidence_interval(slide_accs)
            y_mean_p, y_min_p, y_max_p = generate_confidence_interval(pg_accs)

            if workers == 1:
                legend_slide = 'Single Device SLIDE'
                legend_pg = 'Single Device PGHash-D'
                legend_dense = 'Single Device FedAvg'
            else:
                legend_slide = str(workers) + ' Device SLIDE'
                legend_pg = str(workers) + ' Device PGHash-D'
                legend_dense = str(workers) + ' Device FedAvg'

            plt.figure()
            plt.plot(iters_pg, y_mean_p, label=legend_pg, color='r', alpha=0.8)
            plt.fill_between(iters_pg, y_min_p, y_max_p, alpha=0.2, color='r')

            plt.plot(iters_slide, y_mean_s, label=legend_slide, color='b', alpha=0.8)
            plt.fill_between(iters_slide, y_min_s, y_max_s, alpha=0.2, color='b')

            # dense_file = 'test1-regular-' + dataset + '-' + str(workers) + 'workers-1.0cr'
            # test_acc_d, iters_d = unpack_raw_test(dense_folder + ds + dense_file)
            # plt.plot(iters_d, test_acc_d, label=legend_dense, color='g')

            plt.legend(loc='upper left')
            plt.ylabel('Test Accuracy', fontsize=15)
            plt.xlabel('Iterations', fontsize=15)
            plt.xscale("log")
            plt.grid(which="both", alpha=0.25)
            plt.xlim([100, 1.55e4])
            plt.ylim([0, 0.35])
            # plt.show()
            savefilename = 'amazon' + str(workers) + '-comparison-c8.pdf'
            plt.savefig(savefilename, format="pdf")

    elif dataset == 'Wiki325K':

        for run in range(1, 2):
            for workers in amz_workers:

                plt.figure()
                pg_file = 'run' + str(run) + '-pghash-' + dataset + '-' + str(workers) + 'workers-1.0cr-50tables-50rehash'
                slide_file = 'run' + str(run) + '-slide-' + dataset + '-' + str(workers) + 'workers-1.0cr-50tables-50rehash'

                test_acc_pg, iters_pg = unpack_raw_test(pg_folder + ds + pg_file)
                test_acc_slide, iters_slide = unpack_raw_test(slide_folder + ds + slide_file)

                if workers == 1:
                    legend_slide = 'Single Device SLIDE'
                    legend_pg = 'Single Device PGHash-D'
                else:
                    legend_slide = str(workers) + ' Device SLIDE'
                    legend_pg = str(workers) + ' Device PGHash-D'

                plt.plot(iters_pg, test_acc_pg, label=legend_pg, color='r')
                plt.plot(iters_slide, test_acc_slide, label=legend_slide, color='b')

                plt.legend(loc='upper left')
                plt.ylabel('Test Accuracy', fontsize=15)
                plt.xlabel('Iterations', fontsize=15)
                plt.xscale("log")
                plt.grid(which="both", alpha=0.25)
                # plt.xlim([100, 1.55e4])
                # plt.ylim([0, 0.35])
                plt.show()
                savefilename = 'wiki' + str(workers) + '-comparison-c8.pdf'
                # plt.savefig(savefilename, format="pdf")

    if sampled_softmax:

        for i in range(2):
            ds = datasets[i]

            if ds == 'Amazon670K':
                pg_file = '/run1-pghash-' + ds + '-' + '1workers-1.0cr-50tables-50rehash'
            else:
                pg_file = '/run1-pghash-' + ds + '-' + '1workers-1.0cr-50tables-1rehash'

            plt.figure()
            ss_file = '/sampled-softmax-' + ds + '-' + '1workers-0.1cr'

            test_acc_ss, iters_ss = unpack_raw_test(dense_folder + ds + ss_file)
            test_acc_pg, iters_pg = unpack_raw_test(pg_folder + ds + pg_file)

            plt.plot(iters_ss, test_acc_ss, label='Sampled Softmax', color='k')
            if i == 0:
                plt.plot(iters_pg, test_acc_pg, label='PGHash', color='r')
                plt.legend(loc='lower right')
            else:
                plt.plot(iters_pg, test_acc_pg, label='PGHash-D', color='r')
                plt.legend(loc='upper left')



            plt.ylabel('Test Accuracy', fontsize=15)
            plt.xlabel('Iterations', fontsize=15)
            plt.xscale("log")
            plt.grid(which="both", alpha=0.25)
            if ds == 'Amazon670K':
                plt.xlim([1e2, 1.55e4])
                plt.ylim([0, 0.35])
            else:
                plt.xlim([1e2, 5e3])
                plt.ylim([0.05, 0.48])
            #plt.show()
            savefilename = ds + '-sampled-softmax.pdf'
            plt.savefig(savefilename, format="pdf")

    if avg_neuron:

        # legends = ['PGHash Delicious-200K', 'PGHash Amazon-670K']
        legends = ['PGHash Delicious-200K', 'PGHash-D Amazon-670K']
        colors = ['r', 'b']

        for i in range(2):
            ds = datasets[i]
            pg_neurons = []
            for trial in range(1, ntest + 1):

                if ds == 'Amazon670K':
                    pg_file = '/run' + str(trial) + '-pghash-' + ds + '-' + '1workers-1.0cr-50tables-50rehash'
                    nc = 670091
                    max_iter = 1.6e4
                else:
                    pg_file = '/run' + str(trial) + '-pghash-' + ds + '-' + '1workers-1.0cr-50tables-1rehash'
                    nc = 205443
                    max_iter = 5.1e3

                neurons_pg, iters_pg = unpack_raw_test(pg_folder + ds + pg_file, file_test='r0-avg-active-neurons.log')
                neurons_pg = np.concatenate((neurons_pg[:500], neurons_pg[500:int(max_iter):5]))
                pg_neurons.append(neurons_pg[:int(max_iter)])

            iters = np.concatenate((iters_pg[:500], iters_pg[500:int(max_iter):5]))
            pg_neurons = np.stack(pg_neurons, axis=0)
            y_mean_p, y_min_p, y_max_p = generate_confidence_interval(pg_neurons)

            # plt.figure()

            plt.plot(iters, y_mean_p/nc, label=legends[i], color=colors[i])
            plt.fill_between(iters, y_min_p/nc, y_max_p/nc, alpha=0.2, color=colors[i])

            # plt.plot(iters_pg, neurons_pg/nc, label='PGHash', color='r')

            # plt.plot(iters_pg, neurons_pg, label='PGHash', color='r')
            # plt.plot(iters_pg, nc*np.ones(len(iters_pg)), label='Dense Baseline', color='g')

            plt.legend(loc='upper right')
            plt.ylabel('Average Activated Neurons per Sample (%)', fontsize=15)
            plt.xlabel('Iterations', fontsize=15)
            plt.xscale("log")
            # plt.yscale("log")
            plt.grid(which="both", alpha=0.25)

            #'''
            if ds == 'Amazon670K':
                plt.xlim([1, 1.55e4])
                plt.ylim([0, 0.375])
            else:
                # plt.yscale("log")
                plt.xlim([1, 5e3])
                plt.ylim([0, 0.375])
            #'''
            # plt.show()
            # savefilename = ds + '-log-avg-neurons.pdf'
            savefilename = ds + '-avg-neurons.pdf'
            plt.savefig(savefilename, format="pdf")

