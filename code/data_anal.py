import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_dat(name):
    fil = open('../datasets/tic-tac-toe/runs/' + name, 'r')
    fil.readline()
    ac = float(fil.readline()[:-1])
    fil.readline()
    dat_lin = [float(x) for x in fil.readline()[1:-2].split(",")]
    for i in range(2): fil.readline()
    dat_exp = [float(x) for x in fil.readline()[1:-2].split(",")]
    fil.close()
    return dat_lin, dat_exp, ac


if __name__ == '__main__':
    k = [0, 1, 10, 11, 22]
    kstar = [0, 1, 2, 3]
    f_size = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    tests = [(x, y, z) for x in k for y in kstar for z in f_size]
    for i in f_size:
        tests.remove((0, 0, i))

    big_ass_df = pd.DataFrame()
    for i in tests:
        for j in range(5):
            fil_name = str(i[0]).zfill(2) + "_" + str(i[1]) + "_" + str(i[2]).zfill(4) + "_" + str(
                j).zfill(2)
            dat_l, dat_e, acc = read_dat(fil_name + '_testMetric.txt')
            big_ass_df = big_ass_df.append({'type':'test', 'k':i[0], 'kstar':i[1], 'f_size':i[2], 'acc':acc, 'trial':j,
                                            'l_med':np.median(dat_l), 'l_1st':np.percentile(dat_l, 25), 'l_3rd':np.percentile(dat_l, 75),
                                            'e_med':np.median(dat_e), 'e_1st':np.percentile(dat_e, 25), 'e_3rd':np.percentile(dat_e, 75)},
                                           ignore_index=True)
            dat_l, dat_e, acc = read_dat(fil_name + '_trainMetric.txt')
            big_ass_df = big_ass_df.append(
                {'type': 'train', 'k': i[0], 'kstar': i[1], 'f_size': i[2], 'acc': acc, 'trial': j,
                 'l_med': np.median(dat_l), 'l_1st': np.percentile(dat_l, 25), 'l_3rd': np.percentile(dat_l, 75),
                 'e_med': np.median(dat_e), 'e_1st': np.percentile(dat_e, 25), 'e_3rd': np.percentile(dat_e, 75)},
                ignore_index=True)

    print('shit')
    # plt.errorbar(big_ass_df[big_ass_df['type']=='test']['kstar'], big_ass_df[big_ass_df['type']=='test']['l_med'],
    #              yerr=(np.array(big_ass_df[big_ass_df['type']=='test'][['l_1st', 'l_3rd']]) - np.array(big_ass_df[big_ass_df['type']=='test']['l_med']).reshape([-1,1])).T, fmt='o')
    # plt.show()
    #
    # plt.scatter(big_ass_df[big_ass_df['type']=='test']['l_med'], big_ass_df[big_ass_df['type']=='test']['e_med'], s=1)
    # plt.show()



    dic = {k[i]: i for i in range(len(k))}
    fig = plt.figure()
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=25)
    plt.rc('axes', titlesize=30)
    plt.rc('legend', fontsize=25)
    fig.set_figwidth(12)
    fig.set_figheight(8)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.hlines(0, -1, len(k)+1, color='grey')
    ax.scatter(np.array([dic[i] for i in big_ass_df[big_ass_df['type']=='test'][big_ass_df['f_size']==5000]['k']])-0.1,
               big_ass_df[big_ass_df['type']=='test'][big_ass_df['f_size']==5000]['l_med'], label='linear', s=15)
    ax.scatter(np.array([dic[i] for i in big_ass_df[big_ass_df['type']=='test'][big_ass_df['f_size']==5000]['k']])+0.1,
               big_ass_df[big_ass_df['type'] == 'test'][big_ass_df['f_size']==5000]['e_med'], label='exponential', s=15)
    ax.legend()

    ax.set_xticks(range(len(k)))
    ax.set_xticklabels([0, 1, 10, 11, 22])
    ax.set_xlim(-0.5, len(k)-0.5)
    ax.set_yticks([0.2*x-1 for x in range(11)])
    ax.set_xlabel('Number of Rules (k)')
    ax.set_ylabel('Performance Measure')
    ax.set_title('Against Number of Rules')
    plt.savefig('../Write_up/images/rules_performance_meas.png')

    dic = {f_size[i]: i for i in range(len(f_size))}
    fig = plt.figure()
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=25)
    plt.rc('axes', titlesize=30)
    plt.rc('legend', fontsize=25)
    fig.set_figwidth(12)
    fig.set_figheight(8)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.hlines(0, -1, len(f_size)+1, color='grey')
    ax.scatter(np.array([dic[i] for i in big_ass_df[big_ass_df['type']=='test'][big_ass_df['k']==10]['f_size']])-0.1,
               big_ass_df[big_ass_df['type']=='test'][big_ass_df['k']==10]['l_med'], label='linear', s=30)
    ax.scatter(np.array([dic[i] for i in big_ass_df[big_ass_df['type']=='test'][big_ass_df['k']==10]['f_size']])+0.1,
               big_ass_df[big_ass_df['type'] == 'test'][big_ass_df['k']==10]['e_med'], label='exponential', s=30)
    ax.legend()

    ax.set_xticks(range(len(f_size)))
    ax.set_xticklabels(f_size)
    ax.set_xlim(-0.5, len(f_size)-0.5)
    ax.set_yticks([0.2*x-1 for x in range(11)])
    ax.set_xlabel('Forest Size')
    ax.set_ylabel('Performance Measure')
    ax.set_title('Against Forest Size')
    plt.savefig('../Write_up/images/forest_performance_meas.png')

    pass

