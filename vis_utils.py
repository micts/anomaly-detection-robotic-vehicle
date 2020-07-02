import numpy as  np
import matplotlib.pyplot as plt

def plot_data(data):
    """
        Plot all dataset variables against time ('t'), i.e. as time series.

        # Parameters
        # ----------
        # data : pandas.DataFrame
        #     Dataset as a pandas dataframe.
        """

    subset_ind = [2390, 5504]
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(25, 90))
    for n_column in range(len(data.columns) - 2):

        if data.columns[n_column + 1] == "RxKBTot" or data.columns[n_column + 1] == "TxKBTot":
            y1 = np.log(data.iloc[:subset_ind[0], n_column + 1] + 1)
            y2 = np.log(data.iloc[subset_ind[0]:subset_ind[1], n_column + 1] + 1)
            y3 = np.log(data.iloc[subset_ind[1]:, n_column + 1] + 1)
        else:
            y1 = data.iloc[:subset_ind[0], n_column + 1]
            y2 = data.iloc[subset_ind[0]:subset_ind[1], n_column + 1]
            y3 = data.iloc[subset_ind[1]:, n_column + 1]

        ax[n_column].plot(data.iloc[:subset_ind[0], 0],
                          y1,
                          c='tab:blue',
                          linewidth=3)

        ax[n_column].plot(data.iloc[subset_ind[0]:subset_ind[1], 0],
                          y2,
                          c='tab:red',
                          linewidth=3)

        ax[n_column].plot(data.iloc[subset_ind[1]:, 0],
                          y3,
                          c='tab:orange',
                          linewidth=3)

        ax[n_column].set_xlabel('Time', fontsize=24)
        ax[n_column].set_ylabel(data.columns[n_column + 1], fontsize=24)
        ax[n_column].tick_params(labelsize=20)
    plt.show()


def plot_feature_importances(f_fi, is_lag):

    """
    Plot feature importances produced by random forest, in descending order of importance.
    """

    plt.figure(figsize=(12, 8))
    f = [i[0] for i in f_fi]
    fi = [i[1] for i in f_fi]
    plt.bar(f, fi)
    plt.xlabel('Features', fontsize=19)
    plt.ylabel('Importance', fontsize=19)
    #plt.xticks(rotation=45, horizontalalignment='center')
    rotation = 90 if is_lag else 0
    plt.xticks(rotation=rotation)
    plt.tick_params(labelsize=14)
    plt.title('Ordered feature importances (Random Forest)', fontsize=19)
    if is_lag:
        fig_name = 'feature_importances' + '_lag'
    else:
        fig_name = 'feature_importances'
    plt.savefig('feature_importances/' + fig_name + '.png', bbox_inches='tight')
