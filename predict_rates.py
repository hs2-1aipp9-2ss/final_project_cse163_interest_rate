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

class PredictRates:
    

    def __init__(self, dir: str, country: str) -> pd.DataFrame:
        self._filenames = os.listdir(dir)
        self._country = country
        self._fname_pd_dict: dict[str, pd.DataFrame] = {}

        for filename in self._filenames:
            pathname = os.path.join(dir, filename)
            if os.path.isfile(pathname):
                if os.path.splitext(pathname)[1] == ".csv":
                    self._fname_pd_dict[filename] = pd.read_csv(pathname)
        
 
        cpi = self._fname_pd_dict[f"{self._country}_quarterly_cpi.csv"][["Date", "CPI"]]

        gdp = self._fname_pd_dict[f"{self._country}_quarterly_gdp.csv"][["Date", "GDP per capita"]][:56]

        interest_rate = self._fname_pd_dict[f"{self._country}_quarterly_interest_rate.csv"][["Date", "Interest Rate"]]

        unemployment_rate = self._fname_pd_dict[f"{self._country}_quarterly_unemployment_rate.csv"][["Date", "Unemployment Rate"]][:56]

        merged_zero = gdp.merge(interest_rate, left_on="Date", right_on="Date")
        merged_one = cpi.merge(unemployment_rate, left_on="Date", right_on="Date")
        merged = merged_zero.merge(merged_one, left_on="Date", right_on="Date")
        
        if self._country + "_monthly_stock_index.csv" in os.listdir(dir):
            sorted_stock_index = self._fname_pd_dict[f"{self._country}_monthly_stock_index.csv"]
            sorted_stock_index['Close'] = sorted_stock_index['Close'].astype(str).str.strip()
            sorted_stock_index['Close'] = sorted_stock_index['Close'].astype(str).str.replace(',','')
            sorted_stock_index['Close'] = pd.to_numeric(sorted_stock_index['Close'])
            sorted_stock_index = sorted_stock_index.groupby("Date")["Close"].mean()
            sorted_stock_index = sorted_stock_index.reset_index()
            self._df: pd.DataFrame = pd.merge(sorted_stock_index, merged, on="Date")
        else:
            self._df = merged
        

    def plot_heatmap(self) -> None:
        """
        comment later
        """
        # Build multivariate linkage chart
        sns.pairplot(self._df, height=1.0)
        plt.savefig("Results/" + self._country + "_multivariate_linkage_chart.png")
        plt.close("all")

        # Calculate the correlation coefficient matrix
        # pandas.corr take out NaN value when calculating
        corr = self._df.corr()
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot a heatmap
        # Param: annot=bool, fmt="decimals", cmap="color" 
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", linewidths=.5,
                    vmax=1, vmin=-1, center=0, square=True)
        plt.savefig("Results/" + self._country + "_max_corr.png", facecolor="azure")
        plt.close("all")


    def preprocess_standard(self) -> None:
        """
        Split the dataset for training data and test data and work on
        standardization.
        """
        self._df = self._df.fillna(0)
        features = self._df.loc[:, self._df.columns != "Interest Rate"]
        features = pd.get_dummies(features, drop_first=True)
        label = self._df["Interest Rate"]
        x_train, x_test, self._y_train, self._y_test = \
            train_test_split(features, label, test_size=0.3, random_state=1234)
        
        # Standardization (Z-score normalization) of data
        sc = StandardScaler()
        sc.fit(x_train) # Standardize training data
        self._x_train_std = sc.transform(x_train)
        self._x_test_std = sc.transform(x_test)


    def ridge_regression(self, ALPHA: float=10.0) -> None:
        """
        Predict using Ridge Regression and evaluate its outcomes.
        """
        self._ridge = Ridge(alpha=ALPHA)
        self._ridge.fit(self._x_train_std, self._y_train)

        pred_ridge = self._ridge.predict(self._x_test_std)

        # Evaluation #1: R^2
        # The closer the predicted values are to the observed values,
        # the closer the value of R^2 becomes to 1.
        r2_ridge = r2_score(self._y_test, pred_ridge)

        # Evaluation #2: MAE (Mean Absolute Error)
        # The closer the predicted values are to the observed values, 
        # the smaller MAE.
        # It is said to be less susceptible to outliers as errors are not squared.
        mae_ridge = mean_absolute_error(self._y_test, pred_ridge)

        print("Evaluation: Ridge Regression")
        print(f"R2  : {r2_ridge}")
        print(f"MAE : {mae_ridge}")

        # Regression Coefficient
        print(f"Coef: {self._ridge.coef_}")

        # Scatterplot between predicted and observed data
        plt.xlabel("pred_ridge")
        plt.ylabel("y_test")
        plt.scatter(pred_ridge, self._y_test)
        plt.savefig("Results/" + self._country + "_Ridge Regression")


    def lasso_regression(self, ALPHA: float=.05) -> None:
        """
        """
        lasso = Lasso(alpha=ALPHA)
        lasso.fit(self._x_train_std, self._y_train)

        pred_lasso = lasso.predict(self._x_test_std)

        # Evaluation #1: R^2
        r2_lasso = r2_score(self._y_test, pred_lasso)

        # Evaluation #2: MAE
        mae_lasso = mean_absolute_error(self._y_test, pred_lasso)

        print("Evaluation: Lasso Regression")
        print("R2 : %.3f" % r2_lasso)
        print("MAE : %.3f" % mae_lasso)

        # Regression Coefficient
        print("Coef = ", lasso.coef_)

        # Scatterplot between predicted and observed data
        plt.xlabel("pred_lasso")
        plt.ylabel("y_test")
        plt.scatter(pred_lasso, self._y_test)
        plt.savefig("Results/" + self._country + "_Lasso Regression")


    def elasticnet_regression(self, ALPHA: float=.05) -> None:
        """
        """
        elasticnet = ElasticNet(alpha=ALPHA)
        elasticnet.fit(self._x_train_std, self._y_train)

        pred_elasticnet = elasticnet.predict(self._x_test_std)

        # Evaluation #1: R^2
        r2_elasticnet = r2_score(self._y_test, pred_elasticnet)

        # Evaluation #2: MAE
        mae_elasticnet = mean_absolute_error(self._y_test, pred_elasticnet)

        print("Evaluation: ElasticNet Regression")
        print("R2 : %.3f" % r2_elasticnet)
        print("MAE : %.3f" % mae_elasticnet)

        # Regression Coefficient
        print("Coef = ", elasticnet.coef_)

        # Scatterplot between predicted and observed data
        plt.xlabel("pred_ElasticNet")
        plt.ylabel("y_test")
        plt.scatter(pred_elasticnet, self._y_test)
        plt.savefig("Results/" + self._country + "_ElasticNet Regression")


    def rf_regression(self, ALPHA: float=.05):
        """
        """
        rf = RandomForestRegressor()
        rf.fit(self._x_train_std, self._y_train)

        pred_rf = rf.predict(self._x_test_std)
        # Evaluation #1: R^2
        r2_rf = r2_score(self._y_test, pred_rf)

        # Evaluation #2: MAE
        mae_rf = mean_absolute_error(self._y_test, pred_rf)

        print("Evaluation: Random Forest Regression")
        print("R2 : %.3f" % r2_rf)
        print("MAE : %.3f" % mae_rf)

        # Regression Coefficient
        print("Coef = ", rf.feature_importances_)

        # Scatterplot between predicted and observed data
        plt.xlabel("pred_rf")
        plt.ylabel("y_test")
        plt.scatter(pred_rf, self._y_test)
        plt.savefig("Results/" + self._country + "_Random Forest Regression")