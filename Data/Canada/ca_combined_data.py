import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


def congregate(dir: str) -> pd.DataFrame:
    filenames = os.listdir(dir)
    fname_pd_dict: dict[str, pd.DataFrame] = {}
    for filename in filenames:
        pathname = os.path.join(dir, filename)
        if os.path.isfile(pathname):
            if os.path.splitext(pathname)[1] == ".csv":
                fname_pd_dict[filename] = pd.read_csv(pathname)
    
    sorted_stock_index = fname_pd_dict["ca_monthly_stock_index.csv"]
    sorted_stock_index = sorted_stock_index.groupby("Date")["Close"].mean()
    sorted_stock_index = sorted_stock_index.reset_index()

    
    sorted_cpi = fname_pd_dict["ca_quarterly_cpi.csv"]

    sorted_gdp = fname_pd_dict["ca_quarterly_gdp.csv"][["Date", "GDP per capita"]][:56]

    sorted_interest_rate = fname_pd_dict["ca_quarterly_interest_rate.csv"]

    sorted_unemployment_rate = fname_pd_dict["ca_quarterly_unemployment_rate.csv"][["Date", "Unemployment Rate"]][:56]


    merged_zero = sorted_gdp.merge(sorted_interest_rate, left_on="Date", right_on="Date")
    merged_one = sorted_cpi.merge(sorted_unemployment_rate, left_on="Date", right_on="Date")
    merged = merged_zero.merge(merged_one, left_on="Date", right_on="Date")
    
    finalized = pd.merge(sorted_stock_index, merged, on="Date")


    return finalized


def plot_heatmap(df: pd.DataFrame, dir: str) -> None:
    """
    comment later
    """
    # Build multivariate linkage chart
    sns.pairplot(df, height=1.0)
    plt.savefig(dir + "multivariate_linkage_chart.png")
    plt.close("all")

    # Calculate the correlation coefficient matrix
    # pandas.corr take out NaN value when calculating
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot a heatmap
    # Param: annot=bool, fmt="decimals", cmap="color" 
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", linewidths=.5,
                vmax=1, vmin=-1, center=0, square=True)
    plt.savefig(dir + "ca_max_corr.png", facecolor="azure")
    plt.close("all")


def preprocess_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the dataset for training data and test data and work on
    standardization.
    """
    df = df.fillna(0)
    features = df.loc[:, df.columns != "Interst Rate"]
    features = pd.get_dummies(features, drop_first=True)
    label = df["Interest Rate"]
    x_train, x_test, y_train, y_test = \
        train_test_split(features, label, test_size=0.3, random_state=1234)
    
    # Standardization (Z-score normalization) of data
    sc = StandardScaler()
    sc.fit(x_train) # Standardize training data
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test


def ridge_regression(x_train_std: pd.DataFrame, y_train: pd.DataFrame,
                     x_test_std: pd.DataFrame, y_test: pd.DataFrame,
                     ALPHA: float=10.0) -> None:
    """
    Predict using Ridge Regression and evaluate its outcomes.
    """
    ridge = Ridge(alpha=ALPHA)
    ridge.fit(x_train_std, y_train)

    pred_ridge = ridge.predict(x_test_std)

    # Evaluation #1: R^2
    # The closer the predicted values are to the observed values,
    # the closer the value of R^2 becomes to 1.
    r2_ridge = r2_score(y_test, pred_ridge)

    # Evaluation #2: MAE (Mean Absolute Error)
    # The closer the predicted values are to the observed values, 
    # the smaller MAE.
    # It is said to be less susceptible to outliers as errors are not squared.
    mae_ridge = mean_absolute_error(y_test, pred_ridge)

    print("Evaluation: Ridge Regression")
    print(f"R2  : {r2_ridge}")
    print(f"MAE : {mae_ridge}")

    # Regression Coefficient
    print(f"Coef: {ridge.coef_}")

    # Scatterplot between predicted and observed data
    plt.xlabel("pred_ridge")
    plt.ylabel("y_test")
    plt.scatter(pred_ridge, y_test)
    plt.show()


def lasso_regression(x_train_std: pd.DataFrame, y_train: pd.DataFrame,
                     x_test_std: pd.DataFrame, y_test: pd.DataFrame,
                     ALPHA: float=.05) -> None:
    """
    """
    lasso = Lasso(alpha=ALPHA)
    lasso.fit(x_train_std, y_train)

    pred_lasso = lasso.predict(x_test_std)

    # Evaluation #1: R^2
    r2_lasso = r2_score(y_test, pred_lasso)

    # Evaluation #2: MAE
    mae_lasso = mean_absolute_error(y_test, pred_lasso)

    print("Evaluation: Lasso Regression")
    print("R2 : %.3f" % r2_lasso)
    print("MAE : %.3f" % mae_lasso)

    # Regression Coefficient
    print("Coef = ", lasso.coef_)

    # Scatterplot between predicted and observed data
    plt.xlabel("pred_lasso")
    plt.ylabel("y_test")
    plt.scatter(pred_lasso, y_test)
    plt.show()


def elasticnet_regression(x_train_std: pd.DataFrame, y_train: pd.DataFrame,
                     x_test_std: pd.DataFrame, y_test: pd.DataFrame,
                     ALPHA: float=.05) -> None:
    """
    """
    elasticnet = ElasticNet(alpha=ALPHA)
    elasticnet.fit(x_train_std, y_train)

    pred_elasticnet = elasticnet.predict(x_test_std)

    # Evaluation #1: R^2
    r2_elasticnet = r2_score(y_test, pred_elasticnet)

    # Evaluation #2: MAE
    mae_elasticnet = mean_absolute_error(y_test, pred_elasticnet)

    print("Evaluation: ElasticNet Regression")
    print("R2 : %.3f" % r2_elasticnet)
    print("MAE : %.3f" % mae_elasticnet)

    # Regression Coefficient
    print("Coef = ", elasticnet.coef_)

    # Scatterplot between predicted and observed data
    plt.xlabel("pred_lasso")
    plt.ylabel("y_test")
    plt.scatter(pred_elasticnet, y_test)
    plt.show()


def rf_regression(x_train_std: pd.DataFrame, y_train: pd.DataFrame,
                 x_test_std: pd.DataFrame, y_test: pd.DataFrame,
                 ALPHA: float=.05):
    """
    """
    rf = RandomForestRegressor()
    rf.fit(x_train_std, y_train)

    pred_rf = rf.predict(x_test_std)
    # Evaluation #1: R^2
    r2_rf = r2_score(y_test, pred_rf)

    # Evaluation #2: MAE
    mae_rf = mean_absolute_error(y_test, pred_rf)

    print("Evaluation: ElasticNet Regression")
    print("R2 : %.3f" % r2_rf)
    print("MAE : %.3f" % mae_rf)

    # Regression Coefficient
    print("Coef = ", rf.feature_importances_)

    # Scatterplot between predicted and observed data
    plt.xlabel("pred_rf")
    plt.ylabel("y_test")
    plt.scatter(pred_rf, y_test)
    plt.show()    


def main():
    # Building single DataFrame from different datasets 
    directory = "./Data/Canada/"
    canada_df = congregate(directory)

    # Plotting heatmap and understand the relation between the variables
    plot_heatmap(canada_df, directory)

    # Preprocess of Data
    x_train_std, x_test_std, y_train, y_test = preprocess_standard(canada_df)

    ridge_regression(x_train_std, y_train, x_test_std, y_test, ALPHA=10.0)
    lasso_regression(x_train_std, y_train, x_test_std, y_test, ALPHA=.05)
    elasticnet_regression(x_train_std, y_train, x_test_std, y_test, ALPHA=.05)
    rf_regression(x_train_std, y_train, x_test_std, y_test)

if __name__ == "__main__":
    main() 