# @ AUTHOR : ARDA BAYSALLAR
# VBO DSMLBC-9
# MODUL 3 - MEASURING PROBLEMS - SORTING

# GIVEN STORY OF THE CASE
#######################################################################################################
# One of the most important problems in e-commerce is the correct
# calculation of the points given to the products after sales.
#
# The solution to this problem means providing greater customer
# satisfaction for the e-commerce site, prominence of the product
# for the sellers and a seamless shopping experience for the buyers.
#
# Another problem is the correct ordering of the comments given to the products.
#
# Since misleading comments will directly affect the sale of the product,
# it will cause both financial loss and loss of customers.
#
# In the solution of these 2 basic problems, while the e-commerce site and
# the sellers will increase their sales, the customers will
# complete the purchasing journey without any problems.
#######################################################################################################

# SO BASICS :
# 1 - CORRECT SCORING (MEASURING) OF PRODUCTS AFTER SALES (it will help the ethic and quality
# customer journey for customers)
# ---------------------------------
# 2 - CORECT SORTING OF THE COMMENTS
# GET RID OF FRAUDULENT COMMENTS AND BIAS COMMENTS BECAUSE IT WILL EFFECT THE PRODUCT SALES DIRECTLY


#
# DATA
#######################################################################################################

# reviewerID 	: USER ID
# asin 			: PRODUCT ID
# reviewerName 	: USER NAME
# helpful 		: HELPFULL COMMENT DEGREE
# reviewText 	: REVIEW
# overall 		: PRODUCT RATING
# summary 		: REVIEW SUMMARY
# unixReviewTime : REVIEW TIME
# reviewTime 	: RAW REVIEW TIME
# day_diff 		: DAYS SINCE THE REVIEW
# helpful_yes 	: TOTAL NUMBER OF SUPPORTIVE VOTE FOR REVIEW
# total_vote 	: TOTAL NUMBER OF VOTE FOR REVIEW
#######################################################################################################

#######################################################################################################
# MISSION 1 Calculate the Average Rating according to the current comments and compare it
# with the existing average rating.
#######################################################################################################


# IMPORTS
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DISPLAY SETTINGS
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)
pd.set_option('display.width', 500)


# READING DATA
df_base = pd.read_csv("amazon_review.csv")
df = df_base.copy()


# FUNDAMENTAL ANALYSIS
df.info()

df.head(10)

df.isnull().sum()
df.dropna(inplace=True)

df.head()
# CHECK UNIQUENESS :
df.nunique()
df.shape
# reviewerID is unique
# fun to see that there is only one same comment made :)

# descriptive stats
df.describe().T

# Here we are focused on overall feature which gives the product rating
df.overall.value_counts()
df.overall.value_counts(normalize=True)
# it ranges from 1 - 5 with different ratios
# Biggest volume is with 5 point which is 80 % of the overall scores second is 4 with 11%

df['overall'].mean()

df['overall'].median()

df['overall'].hist(bins=20)
plt.show()

# mean and median show skewness !



# STEP 1: PRODUCT AVG SCORE

avg_score = df.overall.mean()
print('AVERAGE SCORE :', avg_score)


# STEP 2 : AVG SCORE ACCORDING TO DATES
# we have date but we already have days since the last comment we can use it also
# bu we are asked to set dates as datetime format

# df['date'] = pd.to_datetime(df['date'])
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df.info()

current_date = max(df['reviewTime'])
current_date

df['days'] = (current_date - df['reviewTime']).dt.days

df['days'].quantile([1., 0.75, 0.5, 0.25])

df['days'].quantile([0.1, 0.25, 0.50, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 1]).plot()
df['days'].describe([0.1, 0.25, 0.50, 0.75, 0.9, 1])
df['days'].describe([0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.975, 1])


# q1 = 280 gün
# q4 1.00   1063.00
# q3 0.75    600.00
# q2 0.50    430.00
# q1 0.25    280.00


def time_based_weighted_avg(dataframe, w1 = 35, w2=25, w3 =22, w4=18):
    """
    Giving time based weighted average for each comment and give scores accordingly
    :param dataframe: pandas dataframe
    :param w1: q1 weight default 35 % integer
    :param w2: q2 weight default 25 % integer
    :param w3: q3 weight default 22 % integer
    :param w4: q4 weight default 18 % integer
    :return:
    df
    pandas data frame according to weighted avg
    """

    return dataframe.loc[dataframe.days <= 280, 'overall'].mean() * w1/100 + \
           dataframe.loc[dataframe.days > 280, 'overall'].mean() * w2 / 100 + \
           dataframe.loc[dataframe.days > 430 , 'overall'].mean() * w3/100 + \
           dataframe.loc[dataframe.days > 600, 'overall'].mean() * w4 / 100


print('TIME BASED WEIGHTED AVG : ', time_based_weighted_avg(df))

print('NORMAL AVG : ', df.overall.mean())

df['QUARTILES'] = pd.qcut(df['days'], [0., 0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.975, 1],
                          labels=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'])

df[['QUARTILES', 'days', 'overall']].groupby('days').mean().plot.line()
plt.show()

agg_df = df.groupby('QUARTILES').agg({'days': ['mean', 'median'],
                                      'overall': ['mean']})
agg_df.reset_index(inplace=True)
agg_df.hist(bins=10)
from scipy.stats import ttest_1samp, shapiro, mannwhitneyu, f_oneway, levene, kendalltau, ttest_ind,kruskal

agg_df.columns = agg_df.columns.droplevel(1)

agg_df.columns = ['QUARTILES', 'days_mean', 'days_median', 'overall_mean']




# do they distributed normal ?
for g in list(df['QUARTILES'].unique()) :
    pval = shapiro(df.loc[df['QUARTILES'] == g, 'overall'])[1]
    print(g, "p-val : %.4f " % pval)

kruskal(df.loc[df['QUARTILES']=="q1", "overall"],
        df.loc[df['QUARTILES']=="q2", "overall"],
        df.loc[df['QUARTILES']=="q3", "overall"],
        df.loc[df['QUARTILES']=="q4", "overall"],
        df.loc[df['QUARTILES']=="q5", "overall"],
        df.loc[df['QUARTILES']=="q6", "overall"],
        df.loc[df['QUARTILES']=="q7", "overall"],
        df.loc[df['QUARTILES']=="q8", "overall"])

from statsmodels.stats.multicomp import MultiComparison
comparision = MultiComparison(df['overall'], df['QUARTILES'])
tukey  = comparision.tukeyhsd(0.05)
print(tukey.summary())

# NEW QUARTILES ADHUSTED ACCORDING TO KRUSKAL GROUP MEAN DIFFERENCE HYPOTHESIS TESTING:
df['ADJUSTED_QUARTILES'] = pd.qcut(df['days'], [0., 0.75, 0.9, 1],
                                   labels=['q1', 'q2', 'q3'])

# do they distributed normal ?
for g in list(df['ADJUSTED_QUARTILES'].unique()) :
    pval = shapiro(df.loc[df['ADJUSTED_QUARTILES'] == g, 'overall'])[1]
    print(g, "p-val : %.4f " % pval)

# Multi comparision test
kruskal(df.loc[df['ADJUSTED_QUARTILES']=="q1", "overall"],
        df.loc[df['ADJUSTED_QUARTILES']=="q2", "overall"],
        df.loc[df['ADJUSTED_QUARTILES']=="q3", "overall"])

comparision = MultiComparison(df['overall'], df['ADJUSTED_QUARTILES'])
tukey  = comparision.tukeyhsd(0.05)
print(tukey.summary())

# This is a better way to distribute them thanks to the analysis I made above


# TIME BASED WEIGHT AVG CALCULATOR FUNCTION
def time_based_weighted_avg(dataframe, anova_col, target_col,
                            quantile=[0.0, 0.25, 0.5, 0.75, 1.0], *weight_args):
    """
    Giving time based weighted average for each comment and give scores accordingly
    :param dataframe: pandas dataframe
    :param anova_col: the grouping column we will check
    :param target_col: which column to test
    :param quantile: the quantile distribution, default is [0.0, 0.25, 0.5, 0.75, 1.0]
    :param weight_args: weights according to distribution, 1 less then number of quantile
    :return:
    avg : average value
    """

    avg = 0
    for ind, w in enumerate(weight_args):
        treshold = df['days'].quantile(quantile).iloc[ind]
        avg += dataframe.loc[dataframe[anova_col] <= treshold, target_col].mean() * w/100

    return avg

time_based_weighted_avg(df, 'days', 'overall', [0., 0.50, 0.75, 0.9, 1], 40, 25, 20, 15)

# !!! NOTE be careful python is expecting:
# if any keyword argument is used in the function call ü
# then it should always be followed by keyword arguments.

# other weight trials
time_based_weighted_avg(df, 'days', 'overall', [0., 0.50, 0.75, 0.9, 1], 35, 25, 22, 18)
time_based_weighted_avg(df, 'days', 'overall', [0., 0.50, 0.75, 0.9, 1], 50, 20, 17, 13)

# Lets check if the weight of the comment which are more recent have difference than the others
l = {}
for i in range(25, 75, 5):
    l[i] = time_based_weighted_avg(df, 'days', 'overall', [0., 0.50, 0.75, 0.9, 1], i, i-5, round((105-2*i)/2, 0), (105-2*i-int(round((105-2*i)/2, 0))))

import matplotlib.pyplot as plt
plt.ylim(4.67,4.80)
plt.bar(l.keys(), l.values())
plt.xlabel('DAYS')
plt.ylabel('TIME BASED WEIGHTED AVERAGE')
plt.show()

# We are observing since the q1 has higher positive reviews increasing its weight is
# resulting with higher average but the question is to where to stop ????

agg_df = df.groupby('ADJUSTED_QUARTILES').agg({'overall': ['mean']})
agg_df.reset_index(inplace=True)
agg_df['DIF_THAN_AVG'] = agg_df[('overall', 'mean')] - df.overall.mean()

#MISSION 2
# find negative votes
# find scores  (positive-negative, average rating and wilson lower bound)


df.head()

df['helpful_no'] = df['total_vote'] - df['helpful_yes']


# STEP 2 score_pos_neg_diff, score_average_rating and
# wilson_lower_bound scores and add to dataframe



def score_pos_neg_diff(dataframe) :
    return dataframe['helpful_yes']-dataframe['helpful_no']

df['score_pos_neg_diff'] = score_pos_neg_diff(df)

def score_average_rating(up, total):
    if total == 0 :
        return 0
    return up/total

df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],x['total_vote']), axis=1)

# WILSON LOWER BOUND SCORE FUNCTION
# here the positive p is for achieving thumbs up which shows that commenter likes the product.
# phat is representing the proportion of success in Bernouille trial process


def BLW(up, down, CI = 0.95):
    """
    Wilson Lower Bound Scoring.
    Check the advanced explanation :
    https://ardabaysallar.medium.com/sorting-as-a-social-proof-the-sorting-wizard-of-statistics-world-e-b-wilson-3cb10e44401b
    :param up: Positive comment
    :param down: Negative comment
    :param CI: Confidence interval
    :return:
    Score Result
    """
    import math
    import scipy.stats as st

    n = up+down
    if n == 0 :
        return 0
    z = st.norm.ppf(1- (1 - CI) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['wilson_lower_bound'] = df.apply(lambda x: BLW(x['helpful_yes'],x['helpful_no']), axis=1)


df[df['total_vote']>0].sort_values('total_vote', ascending = False)

df.sort_values('wilson_lower_bound', ascending=False).head(20)