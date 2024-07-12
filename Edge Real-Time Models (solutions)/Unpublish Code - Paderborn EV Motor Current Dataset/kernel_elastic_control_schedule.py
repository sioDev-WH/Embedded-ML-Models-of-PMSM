# https://www.kaggle.com/wkirgsn/eda-starter
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py

from datetime import datetime
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Avoid graphics
ENABLE_ANALYSIS = False
ENABLE_CV_PLOT = True
DISPLAY_COEFFICIENTS = True
NUM_TEST_FOLDS = 5
CONTROL_COUNT = 7

'''
According to this introductory paper, id_k1 and iq_k1 are to be treated as target features.
https://arxiv.org/pdf/2003.07273.pdf   (Part I)
Data Set Description: Identifying the Physics Behind an Electric Motor – Data-Driven Learning of the Electrical Behavior 
At the same time, they depend on elementary vectors label-encoded into integers.
'''


def load_data(file):
    raw_data = pd.read_csv(file)
    print("original dataset: ")
    print(raw_data.shape)
    print(raw_data.head())
    # https://stackoverflow.com/a/54999422
    # print(raw_data.agg([min, max]))
    # try
    print(raw_data.groupby(['n_k', 'n_1k']).agg([min, max]).reset_index())
    '''
             id_k          iq_k             epsilon_k   n_k  n_1k      id_k1            iq_k1
        min -2.399998e+02  3.345040e-07     -3.141593    1     1        -287.28080      -11.67833
        max -4.461021e-07  2.399999e+02      3.141592    7     7          71.14584      336.54760
    '''
    if ENABLE_ANALYSIS:
        plt.style.use('seaborn-talk')
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        for c, ax in zip(['n_k', 'n_1k'], axes.flatten()):
            sns.countplot(x=c, data=raw_data, palette="ch:.25", ax=ax)
        pairs = raw_data.assign(pairs=lambda r: r.n_k.astype(str) + '->' + r.n_1k.astype(str))['pairs']
        print(pairs.head())
        print('Transition between elementary vectors count')
        print(pairs.value_counts())

    return raw_data


def partition_data(raw_data):
    # Partition data based on control transitions
    df_set = []
    for nk in range(1, CONTROL_COUNT + 1):
        for n1k in range(1, CONTROL_COUNT + 1):
            df_set.append(raw_data.loc[(raw_data['n_k'] == nk) & (raw_data['n_1k'] == n1k)])
    df_set_cnt = len(df_set)
    print("number of control transition models: ", df_set_cnt)
    return df_set


def build_submodel(dfi, idx):
    if ENABLE_ANALYSIS:
        run_analysis(dfi)

    ##################
    # Feature Engineering
    ##################

    dfi = dfi.assign(sin_eps_k=lambda fdf: np.sin(fdf.epsilon_k),
                     cos_eps_k=lambda fdf: np.cos(fdf.epsilon_k),
                     i_norm=lambda fdf: np.sqrt(fdf.id_k ** 2 + fdf.iq_k ** 2),
                     sin_iq=lambda fdf: (np.sin(fdf.epsilon_k) * fdf.iq_k),
                     sin_id=lambda fdf: (np.sin(fdf.epsilon_k) * fdf.id_k),
                     cos_iq=lambda fdf: (np.cos(fdf.epsilon_k) * fdf.iq_k),
                     cos_id=lambda fdf: (np.cos(fdf.epsilon_k) * fdf.id_k),
                     coscos_idiq=lambda fdf: (np.cos(fdf.epsilon_k) * fdf.id_k * np.cos(fdf.epsilon_k) * fdf.iq_k),
                     cossin_idiq=lambda fdf: (np.sin(fdf.epsilon_k) * fdf.id_k * np.cos(fdf.epsilon_k) * fdf.iq_k),
                     sinsin_idiq=lambda fdf: (np.sin(fdf.epsilon_k) * fdf.id_k * np.sin(fdf.epsilon_k) * fdf.iq_k),
                     exp_iq=lambda fdf: (np.exp(-1.0 * fdf.iq_k * fdf.iq_k)),
                     log_iq=lambda fdf: np.log(fdf.iq_k)) \
        .drop(['n_k', 'n_1k', 'epsilon_k'], axis=1)  # axis=1 to drop columns; axis=0 to drop rows
    print("features data of control model")
    print(dfi.shape)
    print(dfi.head())

    ####################
    # Linear Regression
    ####################
    target_cols = ['id_k1', 'iq_k1']
    input_cols = [c for c in dfi if c not in target_cols]
    cv = KFold(shuffle=True, random_state=2020)

    '''
    Skip Scaling for now
    ---------------------
        ss_y = StandardScaler().fit(df[target_cols])
        df = pd.DataFrame(StandardScaler().fit_transform(df),
                          columns=df.columns)  # actually methodically unsound, but data is large enough
    '''
    # Train, test and score
    '''
        Compare to original kernel results (using one-hot partitioning based on control transitions)
        MSE:
        Scores Mean: 585.6606 A² +- 3.5740 A²
        Scores Min: 583.4754 A², Scores Max: 588.1447 A²
        
        This is a rather weak estimation. Can you beat this score?
    '''
    X, y = dfi[input_cols].values, dfi[target_cols].values

    scores = []
    test_sizes = []
    test_squared_errors = []
    plot_done = False

    for train_idx, test_idx in cv.split(X, y):
        names = dfi.columns
        print("training set idx: ", train_idx, "X: ", X[train_idx].shape, "y: ", y[train_idx].shape)
        robust_scaler = RobustScaler().fit(X[train_idx])
        scaled_X_tr = robust_scaler.fit_transform(X[train_idx])
        scaled_X_tst = robust_scaler.fit_transform(X[test_idx])
        alpha = 0.05
        l1_ratio = 0.5
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        enet_model = enet.fit(scaled_X_tr, y[train_idx])
        # ols = LinearRegression().fit(scaled_X_tr, y[train_idx])
        pred = enet.predict(scaled_X_tst)
        # pred = ss_y.inverse_transform(pred)
        # pred = robust_scaler.inverse_transform(pred)

        # gtruth = ss_y.inverse_transform(y[test_idx])
        gtruth = y[test_idx]  # no scaling
        # gtruth = robust_scaler.inverse_transform(y[test_idx])

        mse = mean_squared_error(pred, gtruth)
        scores.append(mse)
        r2 = r2_score(gtruth, pred)
        print("R2: ", r2)
        sse = mse * len(gtruth)
        test_sizes.append(len(gtruth))
        test_squared_errors.append(sse)
        if ENABLE_CV_PLOT and (not plot_done):
            plot_scatter(gtruth, pred, idx, mse, r2, np.shape(scaled_X_tr)[0], np.shape(scaled_X_tst)[0])
            plot_done = True  # plot for only one test for each model
        if DISPLAY_COEFFICIENTS:
            print("model parameters:\n", enet_model.coef_)
    scores = np.asarray(scores)
    test_sizes = np.asarray(test_sizes)
    test_squared_errors = np.asarray(test_squared_errors)
    print('MSE:')
    print(f'Scores Mean: {scores.mean():.4f} +- {2 * scores.std():.4f}')
    print(f'Scores Min: {scores.min():.4f}, Scores Max: {scores.max():.4f}')
    return [test_sizes, test_squared_errors, scores]


def plot_scatter(measured, predicted, idx, mean, r2, train_size, test_size):
    n_k = int(idx / CONTROL_COUNT) + 1
    n_1k = idx - CONTROL_COUNT * (n_k - 1) + 1
    title = "(n[k]=" + str(n_k) + ", n[k-1]=" + str(n_1k) + ")\n"
    title += str(
        f'mean squared error: {mean:.4f}' + ', R2: ' + f'{r2:.4f}\n' + '(Train: ' + str(train_size) + ', Test: ' + str(
            test_size) + ')\n\n')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Cross Validation - Model[' + str(idx + 1) + ']\n' + title, fontsize=12)  # , fontweight='bold'
    meas = measured[:, 0]
    pred = predicted[:, 0]
    ax1.scatter(meas, pred)
    id_measured_range = [-290, 65]  # hard coding for comparison with Kaggle starter (single-model) kernels
    # id_measured_range = [meas.min(), meas.max()]
    ax1.plot(id_measured_range, id_measured_range, 'k-', lw=1)  # line styles ('-' solid; lw width)
    ax1.set_xlabel('id_meas')
    ax1.set_ylabel('id_pred')
    ax1.set_title("id")
    meas = measured[:, 1]
    pred = predicted[:, 1]
    ax2.scatter(meas, pred)
    iq_measured_range = [-5, 325]  # hard coding for comparison with Kaggle starter (single-model) kernels
    # iq_measured_range = [meas.min(), meas.max()]
    ax2.plot(iq_measured_range, iq_measured_range, 'k-', lw=1)
    ax2.set_xlabel('iq_meas')
    ax2.set_ylabel('iq_pred')
    ax2.set_title("iq")
    # plt.show()
    fig.set_size_inches(10, 7)
    timestr = datetime.now().strftime('_%Y%m%d%H%M%S')
    if idx < 9:
        model_str = '0' + str(idx + 1)
    else:
        model_str = str(idx + 1)
    filename = "SubmodelsScatter_All\\" + "model_" + model_str + timestr + ".png"
    plt.savefig(filename)
    plt.close(fig)
    # plt.pause(1)


def run_analysis(df):
    reduced_data = df.iloc[::1, :]  # ::1000 means select rows located at n*1000 n=0,1,.... from all rows (40963202)
    analyzed_cols = [c for c in df if c != 'n_k']
    unique_elem_vecs = df['n_k'].nunique()
    print("unique n_k = ", unique_elem_vecs)
    fig, axes = plt.subplots(nrows=unique_elem_vecs, ncols=len(analyzed_cols), sharex='col', figsize=(20, 20))
    for k, df in reduced_data.groupby('n_k'):
        for i, c in enumerate(analyzed_cols):
            sns.distplot(df[c], ax=axes[i])
            if i == 0:
                axes[i].set_ylabel(f'n_k = {k}')
    plt.tight_layout()
    print(reduced_data['epsilon_k'].describe())
    corr = reduced_data.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)

    plt.figure(figsize=(14, 14))
    _ = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})


def plot_partitions_errors(models_mse, dataset_mse):
    fig, axes = plt.subplots(1, 1)
    plt.hist(x=models_mse, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('MSE')
    plt.ylabel('No. of Partitions')
    plt.title('MSE of Partitions')
    # plt.show()
    fig.set_size_inches(10, 7)
    title = str(f'Dataset MSE: {dataset_mse.mean():.4f} +- {2 * dataset_mse.std():.4f}\n')
    title += str(f'MSE Min: {dataset_mse.min():.4f}, MSE Max: {dataset_mse.max():.4f}\n\n')
    fig.suptitle('Cross Validation - mean squared error of models\n' + title, fontsize=12)  # , fontweight='bold'
    timestr = datetime.now().strftime('_%Y%m%d%H%M%S')
    filename = "SubmodelsScatter_All\\" + "MSE_Histogram" + timestr + ".png"
    plt.savefig(filename)
    plt.close(fig)


# Main
def build_models(data_file):
    original_data = load_data(data_file)
    dfs = partition_data(original_data)
    number_of_models = len(dfs)
    N = 0
    sse = np.zeros(NUM_TEST_FOLDS)
    models_mse = np.zeros(number_of_models)
    bins = np.arange(15)
    for idx in range(0, number_of_models):
        dfi = dfs[idx]
        print("\n<<<__________________")
        nk = int(idx / CONTROL_COUNT)
        n1k = idx - CONTROL_COUNT * nk
        print('Model[', idx, '](nk =', nk + 1, ', n1k =', n1k + 1, ')')
        print("model df shape = ", dfi.shape)
        [ni, sse_i, scores_i] = build_submodel(dfi, idx)
        N += ni.min()  # max-min is typically  =< 1
        sse = np.add(sse, sse_i)
        models_mse[idx] = scores_i.mean()
        print("__________________>>>\n")
    if N > 0:
        dataset_mse = sse * (1 / N)
    else:
        print("error: test dataset with zero size")

    print('Total MSE:')
    print(f'Total Scores Mean: {dataset_mse.mean():.4f} +- {2 * dataset_mse.std():.4f}')
    print(f'Total Scores Min: {dataset_mse.min():.4f}, Scores Max: {dataset_mse.max():.4f}')
    plot_partitions_errors(models_mse, dataset_mse)
    print(models_mse)


input_file = "Dataset_Electric_Motor.csv"
build_models(input_file)
print("Done!")
