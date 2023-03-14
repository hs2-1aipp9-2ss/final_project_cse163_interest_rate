# Interest Rate and ...? -- An Analysis of Interest Rate and Different Economic Indicators --

Report by Peter Poliakov, Hiromu Sugiyama, Raymond Smith

CSE 163: Intermediate Data Programming

Professor: Hunter Schafer

Project Mentor: Aishah Vakil

The following contents of this file will be the instructions on how to use the code written for our CSE163 Final Project. The full report can be found within this directory named "report.pdf" or at https://docs.google.com/document/d/1c_Fy9Eozfh6fSz_Eqw6qfqcAsWqIBBuWPlPM1tSyj1E/edit?usp=sharing

My presentation on the subject can be found in this video _submit_jt_later_ and the slides in the file "presentation_slides.pptx"

> **Note:** If you happen to only have this README or are missing files, you can get all the required files at https://github.com/hs2-1aipp9-2ss/final_project_cse163_interest_rate.git

## Required Libraries

- os
- Pandas
- Seaborn
- MatPlotLib
- SciKit Learn

> **Installation Line:** pip install pandas/seaborn/atplotlib/sklearn

## Steps for Result Reproduction
1. Check if the data folder has its correct structure
    ```
    Main Directory
    │   README.md
    │   main.py
    │   predict_rates.py
    │   ...
    │
    └───data
    │   │
    │   └───Australia
    │   │   │   australia_monthly_stock_index.csv
    │   │   │   australia_quarterly_cpi.csv
    │   │   │   ...
    │   │   │
    │   └───Canada
    │   │   │   canada_monthly_stock_index.csv
    │   │   │   canada_quarterly_cpi.csv
    │   │   │   ...
    │   │   │
    │   └───Japan
    │   │   │   japan_zone_monthly_stock_index.csv
    │   │   │   ...
    │   │   │
    │   └───...
    └───Results
    │   │   canada_Ridge Regression.png
    │   │   canada_heatmap.png
    │   │   ...
    ```
2. Run predict_rates.py to set up PredictRate object we will use in main.py
3. Run main.py to build object for each country
    - This will take a long time due to necessity to retrieve large amount of data
    - Prints out the evaluations (R^2 and MAE outputs) in the console
      - Evaluation #1: R^2
        - The closer the predicted values are to the observed values,
        - the closer the value of R^2 becomes to 1. 
      - Evaluation #2: MAE (Mean Absolute Error)
        - The closer the predicted values are to the observed values, the smaller MAE.
        - It is said to be less susceptible to outliers as errors are not squared.
    - After running this file, it will automatically add heatmap and regression model visual image in the Results folder. 

> **Note:** For basic tests, just run tests.py

Thanks for taking the time reading about our first Data Science Project
