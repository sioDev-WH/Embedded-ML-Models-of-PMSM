# https://www.kaggle.com/wkirgsn/eda-starter

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
# Ideally install anaconda https://www.anaconda.com/products/individual
# Then install PyCharm for Anaconda https://www.jetbrains.com/pycharm/promo/anaconda/
# For now, install conda (miniconda) at d:\Miniconda3 from https://docs.conda.io/en/latest/miniconda.html
# change to virtual environment that uses Conda instead of pip
# See steps in https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html
# I added a new "conda" via File|Settings|Interpreter
'''
New (conda) environment
Location: D:\PyConds\Miniconda3\envs\Kaggle\ElectricMotors
Python env: 3.7
Conda Executable: D:\PyConds\Miniconda3\condabin\conda.bat
'''
# I had to install the following packages through project Terminal (at the bottom tabs)
# conda install scikit-learn
# conda install seaborn
#
# other packages were available locally as part of miniconda install
# but I still needed to install via File|Settings|Project Interpreter|+

import seaborn as sns  # https://stackoverflow.com/questions/28828917/error-importing-seaborn-module-in-python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

'''
# expected: /kaggle/input/system-identification-of-an-electric-motor/Dataset_Electric_Motor.csv
'''
data_file = "D:\\R&D/Digital Twin\\ElectricMotors\\Models\\Kaggle\\Dataset_Electric_Motor.csv"
'''
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
ENABLE_CV_PLOT = True

plt.style.use('seaborn-talk')

def plot_scatter(measured, predicted):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    meas = measured[:, 0]
    pred = predicted[:, 0]
    ax1.scatter(meas, pred)
    ax1.plot([meas.min(), meas.max()], [meas.min(), meas.max()], 'k-', lw=1)  # line styles ('-' solid; lw width)
    ax1.set_xlabel('id_meas')
    ax1.set_ylabel('id_pred')
    meas = measured[:, 1]
    pred = predicted[:, 1]
    ax2.scatter(meas, pred)
    ax2.plot([meas.min(), meas.max()], [meas.min(), meas.max()], 'k-', lw=1)
    ax2.set_xlabel('iq_meas')
    ax2.set_ylabel('iq_pred')
    plt.show()


# #expected: (40963202, 7)
dataset = pd.read_csv(data_file)
print(dataset.shape)


'''
Expected:
id_k	iq_k	epsilon_k	n_k	n_1k	id_k1	iq_k1
0	-81.45802	229.52930	2.240254	5	6	-105.73820	167.36170
1	-140.68210	112.42340	-1.610116	7	2	-174.89710	128.22370
2	-127.07240	171.74380	-1.971891	4	7	-92.96102	126.60810
3	-42.27880	120.14950	1.300341	2	7	-82.23310	124.33790
4	-48.02003	10.97132	-1.778834	1	4	-45.73148	11.60761'''
print(dataset.head())


'''
Distribution
According to this introductory paper, id_k1 and iq_k1 are to be treated as target features.
https://arxiv.org/pdf/2003.07273.pdf   (Part I)
Data Set Description: Identifying the Physics Behind an Electric Motor – Data-Driven Learning of the Electrical Behavior 
At the same time, they depend on elementary vectors label-encoded into integers.

Let's analyze how they are distributed.
'''
fig, axes = plt.subplots(1,2,sharex=True, sharey=True)
for c, ax in zip(['n_k', 'n_1k'], axes.flatten()):
    sns.countplot(x=c, data=dataset, palette="ch:.25", ax=ax)
unique_elem_vecs = dataset['n_k'].nunique()


'''
Element vector with k = 1 appears significantly more often than the remaining element vector types.
Expected:
0    5->6
1    7->2
2    4->7
3    2->7
4    1->4
Name: pairs, dtype: object
'''
pairs = dataset.assign(pairs=lambda r: r.n_k.astype(str)+'->'+r.n_1k.astype(str))['pairs']
pairs.head()

'''
Transition between elementary vectors count
1->1    2210498
4->5    1272356
5->4    1261304
6->3    1256704
2->7    1256269
3->6    1254441
7->2    1247070
1->4    1061976
2->1    1053721
7->1    1053578
4->1    1052217
6->1    1049703
1->2    1047993
5->1    1042593
1->3    1040018
3->1    1039340
1->6    1039028
1->7    1031871
1->5    1015482
4->4     737755
7->7     723542
5->5     718738
6->6     717718
3->3     706984
2->2     706595
6->7     661480
4->6     654471
5->3     648669
3->2     640841
7->4     636526
2->5     634585
3->4     602125
7->6     599525
4->2     599094
6->4     597456
7->3     594975
6->5     593412
2->6     593304
3->5     589720
5->7     588547
2->3     586896
4->7     578561
5->2     577392
5->6     573614
7->5     567789
2->4     563738
3->7     563684
4->3     560923
6->2     558381
Name: pairs, dtype: int64
'''
print('Transition between elementary vectors count')
pairs.value_counts()



reduced_data = dataset.iloc[::1000, :]
analyzed_cols = [c for c in dataset if c != 'n_k']
fig, axes = plt.subplots(nrows=unique_elem_vecs, ncols=len(analyzed_cols), sharex='col', figsize=(20, 20))

for k, df in reduced_data.groupby('n_k'):
    for i, c in enumerate(analyzed_cols):
        sns.distplot(df[c], ax=axes[k-1, i])
        if i == 0:
            axes[k-1, i].set_ylabel(f'n_k = {k}')
plt.tight_layout()
'''
It becomes evident that certain transitions in the elementary vectors are more common than others.

Moreover, depending on the current elementary vector, distribution of currents and rotor angle epsilon_k is 
either unimodal or bimodal distributed.

More subtle, we recognize a semi-sphere shape of the 2d histogram between the currents (remember, d and q currents 
are to be plotted perpendicular to each other).

It might be auspicious, to add another feature denoting the current vector norm id^2 + iq^2.

On another note, epsilon is the rotor angle, which by design has a value discontinuity in the extreme points over time.
'''


'''
count    40964.000000
mean         0.018398
std          1.808303
min         -3.141521
25%         -1.551863
50%          0.045979
75%          1.584972
max          3.141281
Name: epsilon_k, dtype: float64
'''
reduced_data['epsilon_k'].describe()
'''
Obviously, the value range is clipped to [- pi ,  pi ].

As many ML methods do not respond well to discontinuities in the input space with no corresponding effect on the target space, we replace epsilon by its sine and cosine.
'''

##################
# Feature Engineering
##################
'''
We add sine and cosine of the rotor angle and the current vector norm.

id_k	iq_k	n_k	n_1k	id_k1	iq_k1	sin_eps_k	cos_eps_k	i_norm
0	-81.45802	229.52930	5	6	-105.73820	167.36170	0.784158	-0.620561	243.555145
1	-140.68210	112.42340	7	2	-174.89710	128.22370	-0.999227	-0.039310	180.084630
2	-127.07240	171.74380	4	7	-92.96102	126.60810	-0.920634	-0.390426	213.642991
3	-42.27880	120.14950	2	7	-82.23310	124.33790	0.963649	0.267170	127.371108
4	-48.02003	10.97132	1	4	-45.73148	11.60761	-0.978438	-0.206540	49.257417
'''
dataset = dataset.assign(sin_eps_k=lambda df: np.sin(df.epsilon_k),
                         cos_eps_k=lambda df: np.cos(df.epsilon_k),
                         i_norm=lambda df: np.sqrt(df.id_k**2 + df.iq_k**2)).drop('epsilon_k', axis=1)
dataset.head()

####################
#Correlation Matrix
####################
'''
We observe strong linear correlation between consecutive current measurements in d/q coordinates each.

All other pair-wise comparisons are relatively uncorrelated.
'''
corr = reduced_data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center="dark", as_cmap=True)

plt.figure(figsize=(14,14))
_ = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

####################
# Linear Regression
####################
'''
We kick off regression with a linear model, as the correlation matrix suggests expedient estimation performance just from actual currents. Since elementary vectors are to be treated as categorical, we one-hot encode them before training.

Moreover, in order to fit in RAM, we subsample the data.
'''
df = dataset.iloc[::100, :]\
            .assign(**{**{f'n_k_{i}': lambda x: (x.n_k == i).astype(int) for i in range(1, 8)},
                       **{f'n_1k_{i}': lambda x: (x.n_1k == i).astype(int) for i in range(1, 8)}})\
            .drop(['n_k', 'n_1k'], axis=1)

target_cols = ['id_k1', 'iq_k1']
input_cols = [c for c in df if c not in target_cols]
cv = KFold(shuffle=True, random_state=2020)

ss_y = StandardScaler().fit(df[target_cols])
df = pd.DataFrame(StandardScaler().fit_transform(df),
                     columns=df.columns)  # actually methodically unsound, but data is large enough


# Train, test and score
'''
MSE:
Scores Mean: 585.6606 A² +- 3.5740 A²
Scores Min: 583.4754 A², Scores Max: 588.1447 A²

This is a rather weak estimation. Can you beat this score?
'''
X, y = df[input_cols].values, df[target_cols].values

scores = []
plot_done = False
for train_idx, test_idx in cv.split(X, y):
    ols = LinearRegression().fit(X[train_idx], y[train_idx])
    pred = ols.predict(X[test_idx])
    pred = ss_y.inverse_transform(pred)
    gtruth = ss_y.inverse_transform(y[test_idx])
    scores.append(mean_squared_error(pred, gtruth))
    if ENABLE_CV_PLOT and (not plot_done):
        plot_scatter(gtruth, pred)
        plot_done = True  # plot for only one test for each model

scores = np.asarray(scores)
print('MSE:')
print(f'Scores Mean: {scores.mean():.4f} A² +- {2*scores.std():.4f} A²\nScores Min: {scores.min():.4f} A², Scores Max: {scores.max():.4f} A²')

