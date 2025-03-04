# MoDeVa Data Insights Explorer

## Description

Welcome to the MoDeVa Data Insights Explorer! This Streamlit application provides an interactive demonstration of data analysis and feature engineering techniques, similar to those available in MoDeVa, a powerful data analysis and model development tool.

This application allows you to:

- **Load Data:** Explore insights using a synthetic dataset or upload your own CSV file.
- **Perform Basic Operations:** View summary statistics of your dataset for a quick overview.
- **Conduct Exploratory Data Analysis (EDA):**
    - **Univariate Analysis (eda_1d):** Visualize the distribution of individual numeric features using histograms and density plots.
    - **Bivariate Analysis (eda_2d):** Explore relationships between pairs of features using scatter plots (for numeric vs. numeric features) and box plots (for numeric vs. categorical features).
- **Apply Data Processing and Feature Engineering Techniques:**
    - **Numerical Feature Scaling (scale_numerical):** Experiment with different scaling methods (Standardize, MinMax, Log1p, Quantile) to transform numeric features.
    - **Categorical Feature Encoding (encode_categorical):** Encode categorical features using One-Hot Encoding or Ordinal Encoding.
    - **Numerical Feature Binning (bin_numerical):** Discretize numeric features into bins using Uniform or Quantile binning methods.

This application is designed for educational purposes to illustrate data manipulation and visualization concepts. It provides a hands-on experience with techniques commonly used in data science workflows.

## Installation

To run this Streamlit application, you need to have Python installed on your system, along with pip, the Python package installer. Follow these steps to set up the environment:

1.  **Install Python:** If you don't have Python installed, download and install the latest version from the official Python website: [https://www.python.org/](https://www.python.org/)

2.  **Install required Python packages:** Open your terminal or command prompt and install the necessary libraries using pip:

    ```bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn
    ```

    This command will install:
    - `streamlit`: For creating the interactive web application.
    - `pandas`: For data manipulation and analysis.
    - `numpy`: For numerical operations.
    - `matplotlib`: For plotting.
    - `seaborn`: For enhanced data visualizations.
    - `scikit-learn`: For QuantileTransformer used in feature scaling simulation.

## Usage

1.  **Save the Streamlit application code:** Copy the provided Python code (the code block you provided in the prompt) and save it as a Python file, for example, `app.py`.

2.  **Run the application:** Open your terminal or command prompt, navigate to the directory where you saved `app.py`, and run the Streamlit application using the following command:

    ```bash
    streamlit run app.py
    ```

3.  **Interact with the application:** Streamlit will automatically open the application in your web browser. You can now interact with the application through the user interface:

    - **Sidebar:** Located on the left, the sidebar contains the MoDeVa logo, application title "MoDeVa Insights", and a brief description.

    - **Data Loading Section:**
        - Choose between using a "Synthetic Dataset" or uploading your own "CSV" file using the radio buttons.
        - If you choose "Upload CSV", use the file uploader to select a CSV file from your local machine.
        - A preview of the loaded dataset (first few rows) will be displayed.

    - **Basic Operations Section:**
        - View the summary statistics of the dataset, similar to MoDeVa's `DataSet.summary` function, providing descriptive statistics for numeric columns.

    - **Exploratory Data Analysis (EDA) Section:**
        - **Univariate Analysis (eda_1d):** Select a numeric feature from the dropdown to visualize its distribution using a histogram and density plot.
        - **Bivariate Analysis (eda_2d):** Select up to two features to explore their relationship.
            - For two numeric features, a scatter plot will be displayed.
            - For a numeric and a categorical feature, a box plot will be shown.

    - **Data Processing and Feature Engineering Section:**
        - **Numerical Feature Scaling (scale_numerical):**
            - Choose a numeric feature to scale.
            - Select a scaling method (standardize, minmax, log1p, quantile) from the dropdown.
            - Observe the scaled distribution of the feature through a histogram and density plot.
        - **Categorical Feature Encoding (encode_categorical):**
            - If categorical columns are present, choose a categorical feature to encode.
            - Select an encoding method (onehot, ordinal).
            - View the first few rows of the encoded dataset.
        - **Numerical Feature Binning (bin_numerical):**
            - Choose a numeric feature to bin.
            - Use the slider to select the desired number of bins.
            - Select a binning method (uniform, quantile).
            - View the first few rows of the dataset with the binned feature.

    - **Footer:** The footer contains copyright information and a caption stating the educational purpose of the demonstration and reproduction restrictions.

## Credits

This application is brought to you by **QuantUniversity**.

- **QuantUniversity:** For providing the concept and branding for this educational demonstration.
- **Streamlit:** For the powerful and easy-to-use framework that makes this interactive application possible.
- **Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn:** For the essential data science libraries used in this application.

## License

© 2025 QuantUniversity. All Rights Reserved.

This demonstration is provided solely for educational use and illustration. To access the full legal documentation, please visit [link to legal documentation if available, otherwise remove this part]. Any reproduction of this demonstration requires prior written consent from QuantUniversity.
