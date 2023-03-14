'''
Peter Poliakov, Hiromu Sugiyama, Raymond Smith
CSE 163
This file contains the PredictRates class, which takes data
of 4 different economic indicators: GDP, unemployment, stock,
and CPI and predicts changing interest rates.
'''
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


class PredictRates:

    def __init__(self, dir: str, country: str) -> pd.DataFrame:
        '''
        Initializes a PredictRates object. Takes a str containing
        the directory and a str containing the country and merges them
        by date.
        '''
        self._filenames = os.listdir(dir)
        self._country = country
        self._fname_pd_dict: dict[str, pd.DataFrame] = {}
        for filename in self._filenames:
            pathname = os.path.join(dir, filename)
            if os.path.isfile(pathname):
                if os.path.splitext(pathname)[1] == ".csv":
                    self._fname_pd_dict[filename] = pd.read_csv(pathname)

        cpi = self._fname_pd_dict[f"{self._country}_quarterly_cpi.csv"][
            ["Date", "CPI"]]

        gdp = self._fname_pd_dict[f"{self._country}_quarterly_gdp.csv"][
            ["Date", "GDP per capita"]][:56]

        interest_rate = self._fname_pd_dict[
            f"{self._country}_quarterly_interest_rate.csv"][
            ["Date", "Interest Rate"]]
        interest_rate = interest_rate.groupby("Date")["Interest Rate"].mean()
        unemployment_rate = self._fname_pd_dict[
            f"{self._country}_quarterly_unemployment_rate.csv"][
            ["Date", "Unemployment Rate"]][:56]

        merged_zero = gdp.merge(interest_rate, left_on="Date", right_on="Date")
        merged_ = cpi.merge(unemployment_rate, left_on="Date", right_on="Date")
        merged = merged_zero.merge(merged_, left_on="Date", right_on="Date")

        check_stock = self._country + "_monthly_stock_index.csv"
        if check_stock in os.listdir(dir):
            sorted_stock_index = self._fname_pd_dict[
                f"{self._country}_monthly_stock_index.csv"]
            sorted_stock_index['Close'] = sorted_stock_index[
                'Close'].astype(str).str.strip()
            sorted_stock_index['Close'] = sorted_stock_index[
                'Close'].astype(str).str.replace(',', '')
            sorted_stock_index['Close'] = pd.to_numeric(
                sorted_stock_index['Close'])
            sorted_stock_index = sorted_stock_index.groupby(
                "Date")["Close"].mean()
            sorted_stock_index = sorted_stock_index.reset_index()
            self._df: pd.DataFrame = pd.merge(
                sorted_stock_index, merged, on="Date")
        else:
            self._df = merged

    def plot_heatmap(self) -> None:
        """
        Plots a heatmap showing how different economic indicators affect
        interest rates.
        """

        # Calculate the correlation coefficient matrix
        # pandas.corr take out NaN value when calculating
        corr = self._df.corr()
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot a heatmap
        # Param: annot=bool, fmt="decimals", cmap="color"
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", linewidths=.5,
                    vmax=1, vmin=-1, center=0, square=True)
        plt.title(self._country + " Heatmap")
        plt.savefig("Results/" + self._country + "_heatmap.png",
                    facecolor="azure")
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

        x_train, x_test, self._y_train, self._y_test = train_test_split(
            features, label, test_size=0.35714286, shuffle=False)

        # Standardization (Z-score normalization) of data
        sc = StandardScaler()
        sc.fit(x_train)
        self._x_train_std = sc.transform(x_train)
        self._x_test_std = sc.transform(x_test)

    def ridge_regression(self, ALPHA: float = 10.0) -> None:
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
        mae_ridge = mean_absolute_error(self._y_test, pred_ridge)

        print("Evaluation: Ridge Regression")
        print(f"R2  : {r2_ridge}")
        print(f"MAE : {mae_ridge}")

        # Regression Coefficient
        print(f"Coef: {self._ridge.coef_}")

        # Scatterplot between predicted and observed data
        fig, ax = plt.subplots(1)

        after_2017 = self._df['Date'].str[0:4].astype(int) > 2017
        test_data = self._df[after_2017]
        print(test_data['Date'])
        print(pred_ridge)

        if len(test_data) < len(pred_ridge):
            dif = len(pred_ridge) - len(test_data)
            pred_ridge = pred_ridge[:-dif]
        elif len(test_data) > len(pred_ridge):
            dif = len(test_data) - len(pred_ridge)
            test_data = test_data[:-dif]

        plt.xlabel("Quarter")
        plt.ylabel("Interest Rate")
        plt.title(self._country +
                  " Predicted Interest Rates vs Real Interest Rates")
        plt.plot(test_data['Date'], pred_ridge, c='Blue', label="Prediction")
        plt.plot(test_data['Date'], test_data['Interest Rate'],
                 c='Red', label="Real")
        plt.xticks(rotation=-45)
        plt.legend(loc="upper left")
        plt.savefig("Results/" + self._country + "_Ridge Regression")
