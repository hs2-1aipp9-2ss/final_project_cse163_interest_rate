'''
Peter Poliakov, Hiromu Sugiyama, Raymond Smith
CSE 163
This program creates 8 PredictRates objects, each for Canada, Australia,
Japan, United States, United Kingdom, South Africa, South Korea, or the
Euro Zone, respectively. PredictRates allows us to create predictions on
how certain economic indicators (GDP, unemployment rate, CPI, stock) affect
interest rates.
'''
from predict_rates import PredictRates as pr


def create_pr(countries: list[str]) -> None:
    '''
    Takes a list of countries and creates PredictRates for each,
    including a heatmap and regression model.
    '''
    pred = {}
    for country in countries:
        dir = "./Data/" + country.capitalize().replace('_', ' ') + "/"
        pred[country] = pr(dir, country)
        # Plotting heatmap and understand the relation between the variables
        pr.plot_heatmap(pred[country])

        # Preprocess of Data
        pr.preprocess_standard(pred[country])

        pr.ridge_regression(pred[country], ALPHA=10.0)


def main():
    countries = ['canada', 'australia', 'japan', 'united_states',
                 'united_kingdom', 'south_africa', 'south_korea', 'euro_zone']
    create_pr(countries)


if __name__ == "__main__":
    main()