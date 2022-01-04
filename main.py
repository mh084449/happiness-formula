import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# loading the data
fullData = pd.read_csv("2018.csv")
# clearing out unnecessary information
fullData = fullData.drop(["Overall rank"], axis=1)
fullData.at[19, 'Country or region'] = "Philistine"
fullData = fullData.set_index("Country or region")
# sampling
happiness_data = fullData.sample(30)


# delivering the basic descriptive analysis of the dataset
def descripeData():
    d_data = pd.DataFrame([happiness_data.mean(), happiness_data.median(), happiness_data.var(), happiness_data.std()],
                          index=['Mean', 'Median', 'Variance', 'Standard deviation'])
    return d_data


def estimateSingleVals(name=None, value=None):
    pear_r = pd.Series(data=[happiness_data["Score"].corr(happiness_data[c]) for c in list(happiness_data.columns)[1:]],
                       index=list(happiness_data.columns)[1:])
    if name is not None and value is not None:
        b_1 = pear_r[name] * (descripeData()[name][3] / descripeData()["Score"][3])
        b_0 = descripeData()[name][0] - b_1 * descripeData()["Score"][0]
        estimated_val = float(b_0) + float(b_1) * float(value)
        return estimated_val
    else:
        return pear_r


def inference():

    largest_population_mean = happiness_data.mean() + 1.96 * happiness_data.std() / math.sqrt(30)
    smallest_population_mean = happiness_data.mean() - 1.96 * happiness_data.std() / math.sqrt(30)
    estimated_population_mean = ["{} < x ({}) < {}".format("%.3f" % smallest_population_mean[i],
                                                            i,
                                                           "%.3f" % largest_population_mean[i])
                                 for i in list(happiness_data.columns)]
    return estimated_population_mean


def plotData(y_vals):
    pred_vals = happiness_data[:].copy()
    happiness_data.plot(kind='pie', subplots=True, labeldistance=None, layout=(2, 4))
    happiness_data.plot(kind='bar', subplots=True, layout=(2, 4), fontsize=6)
    for y, c in zip(list(happiness_data.columns)[1:], ['c', 'k', 'y', 'r', 'g', 'b']):
        d = np.polyfit(happiness_data["Score"], happiness_data[y], 1)
        f = np.poly1d(d)
        pred_vals.insert(7, "pred_y_" + y, f(happiness_data["Score"]))
        ax = pred_vals.plot(x="Score", y="pred_y_" + y, color="c")
        happiness_data.plot(kind='scatter',
                            x="Score",
                            y=y,
                            color=c,
                            title="Pearson's r = {}".format("%.4f" % estimateSingleVals()[y]),
                            ax=ax,
                            label='Score vs {}'.format(y)
                            )

    d = np.polyfit(y_vals[0]['Actual Values'], y_vals[0]['Predicted Value'], 1)
    f = np.poly1d(d)
    y_vals[0].insert(3, "y_reged", f(y_vals[0]['Actual Values']))
    ax = y_vals[0].plot(x="Actual Values", y="y_reged", color="c")
    y_vals[0].plot(x='Actual Values',
                   kind="scatter",
                   y='Predicted Value',
                   ax=ax,
                   title='Actual vs Predicted (r2 = {})'.format("%.3f" % y_vals[1]))
    plt.show()


def multipleLinearRegs():
    x = fullData.drop(['Score'], axis=1).values
    y = fullData["Score"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    ml = LinearRegression()
    ml.fit(x_train, y_train)
    y_pred = ml.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    pred_y_df = pd.DataFrame({'Actual Values': y_test, 'Predicted Value': y_pred, 'Difference': y_test - y_pred})
    return pred_y_df, r2


# multipleLinearRegs()
happiness_data.to_csv(r'rand_sample_data.csv')
print("descriptive data about the countries from which the data collected:\n", descripeData())
ans = input("do want to predict the happiness score based on a data you have? (y/n)")
plotData(multipleLinearRegs())
if ans.upper() == 'Y':
    while True:
        try:
            str = "/".join(list(happiness_data.columns)[1:7])
            value_type = input("What kind of data u have? ({})".format(str))
            value = input("\nEnter the data plz: ")
            print("Here's the happiness score: {0:.3f}".format(estimateSingleVals(value_type, value)))
            break
        except:
            print("\nplz enter the name exactly as shown")
if input("do u want to infer the population mean? ").upper() == "Y":
    print("\n".join(inference()))
    print(fullData.mean())