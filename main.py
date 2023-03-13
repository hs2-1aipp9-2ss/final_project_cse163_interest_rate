from predict_rates import PredictRates as pr
import os


def create_pr(countries: list[str]) -> None:
     pred = {}
     for country in countries:
        dir = "./Data/" + country + "/"
        pred[country] = pr(dir, country) 
    
        # Plotting heatmap and understand the relation between the variables
        pr.plot_heatmap(pred[country])

        # Preprocess of Data
        pr.preprocess_standard(pred[country])

        pr.ridge_regression(pred[country], ALPHA=10.0)
        pr.lasso_regression(pred[country], ALPHA=.05)
        pr.elasticnet_regression(pred[country], ALPHA=.05)
        pr.rf_regression(pred[country])

def main():
    countries = ['Canada']
    create_pr(countries)

if __name__ == "__main__":
    main() 