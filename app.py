import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="MoDeVa Data Insights Explorer", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("MoDeVa Insights")
st.sidebar.markdown("Explore MoDeVa's data insights capabilities interactively.")
st.sidebar.divider()

st.title("MoDeVa Data Insights Explorer")
st.markdown("## Overview")
st.write(
    "Welcome to the MoDeVa Data Insights Explorer! This application demonstrates key data operations and visualizations "
    "available in MoDeVa, a powerful tool for data analysis and model development. "
    "Explore the interactive features below to understand data manipulation, exploratory data analysis, and feature engineering concepts."
)
st.divider()

# --- Synthetic Dataset Creation ---
@st.cache_data
def create_synthetic_dataset():
    """Generates a synthetic dataset for demonstration purposes."""
    np.random.seed(42)
    n_samples = 200
    data = {
        'Numeric_Feature_1': np.random.rand(n_samples) * 100,
        'Numeric_Feature_2': np.random.randn(n_samples) * 20 + 50,
        'Categorical_Feature_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'Categorical_Feature_2': np.random.choice(['High', 'Medium', 'Low'], n_samples),
        'Time_Series_Feature': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'Target_Variable': np.random.rand(n_samples) * 50 + 10  # Example Target
    }
    return pd.DataFrame(data)

synthetic_df = create_synthetic_dataset()

# --- Data Loading Section ---
st.header("1. Data Loading")
st.subheader("Dataset Source")
st.markdown(
    "This application uses a **synthetic dataset** for demonstration. It mimics real-world data with numeric, categorical, and time-series features. "
    "You can also **upload your own CSV dataset** to explore its insights using MoDeVa-like operations."
)

dataset_option = st.radio("Choose Dataset:", ["Synthetic Dataset", "Upload CSV"], index=0)

if dataset_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            df = None  # Ensure df is None if loading fails
    else:
        df = None  # df is None if no file is uploaded yet
        st.info("Upload a CSV file to proceed.")

else: # Synthetic Dataset
    df = synthetic_df
    st.info("Using synthetic dataset for demonstration.")

if df is not None: # Proceed if df is loaded successfully (either synthetic or uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.divider()

    # --- Basic Operations Section ---
    st.header("2. Basic Operations")
    st.subheader("Summary Statistics")
    st.markdown(
        "The `DataSet.summary` function in MoDeVa provides a comprehensive overview of your data. "
        "Below is a summary of the dataset, including descriptive statistics for numerical features and value counts for categorical features."
    )
    st.dataframe(df.describe())
    st.markdown(
        "**Explanation:** This summary table provides key statistical measures for numeric columns like mean, standard deviation, min, max, and quartiles. "
        "For categorical columns, it would typically show counts and unique values (though not directly in `df.describe()`, but MoDeVa's `DataSet.summary` would)."
    )

    st.divider()

    # --- EDA Section ---
    st.header("3. Exploratory Data Analysis (EDA)")

    st.subheader("Univariate Analysis (eda_1d)")
    st.markdown(
        "Univariate analysis helps understand the distribution of individual features. "
        "MoDeVa's `DataSet.eda_1d` function generates histograms and density plots for numeric features. "
        "Select a numeric feature below to visualize its distribution."
    )
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        feature_1d_choice = st.selectbox("Select Numeric Feature for Univariate Analysis:", numeric_columns)
        if feature_1d_choice:
            st.markdown(f"### Distribution of {feature_1d_choice}")
            fig_1d, ax_1d = plt.subplots(figsize=(10, 4))
            sns.histplot(df[feature_1d_choice], kde=True, ax=ax_1d)
            ax_1d.set_title(f'Histogram and Density Plot of {feature_1d_choice}')
            ax_1d.set_xlabel(feature_1d_choice)
            ax_1d.set_ylabel('Frequency')
            st.pyplot(fig_1d)
            st.markdown(
                f"**Explanation:** The histogram visualizes the frequency distribution of values for the selected feature, {feature_1d_choice}. "
                "The overlaid density plot provides a smoothed estimate of the probability density function, showing the shape of the data's distribution."
            )
    else:
        st.warning("No numeric columns available for univariate analysis in the uploaded dataset.")

    st.divider()

    st.subheader("Bivariate Analysis (eda_2d)")
    st.markdown(
        "Bivariate analysis explores relationships between pairs of features. "
        "MoDeVa's `DataSet.eda_2d` function can create scatter plots, heatmaps, and boxplots. "
        "Select two features below to visualize their relationship."
    )
    all_columns = df.columns.tolist()
    feature_2d_choices = st.multiselect("Select Two Features for Bivariate Analysis:", all_columns, max_selections=2)

    if len(feature_2d_choices) == 2:
        feature_x, feature_y = feature_2d_choices

        st.markdown(f"### Bivariate Analysis: {feature_x} vs {feature_y}")

        # --- Scatter Plot ---
        if df[feature_x].dtype in np.number and df[feature_y].dtype in np.number:
            st.markdown("#### Scatter Plot")
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=feature_x, y=feature_y, data=df, ax=ax_scatter)
            ax_scatter.set_title(f'Scatter Plot: {feature_x} vs {feature_y}')
            ax_scatter.set_xlabel(feature_x)
            ax_scatter.set_ylabel(feature_y)
            st.pyplot(fig_scatter)
            st.markdown(
                "**Explanation:** A scatter plot visualizes the relationship between two numeric features. "
                "Each point represents a data sample, with its position determined by its values for the selected features. "
                "Patterns in the scatter plot can indicate correlation or other relationships between the features."
            )

        # --- Box Plot (if one is numeric and one is categorical) ---
        elif (df[feature_x].dtype in np.number and df[feature_y].dtype == 'object') or \
             (df[feature_y].dtype in np.number and df[feature_x].dtype == 'object'):
            st.markdown("#### Box Plot")
            numeric_feature = feature_x if df[feature_x].dtype in np.number else feature_y
            categorical_feature = feature_y if df[feature_x].dtype in np.number else feature_x

            fig_boxplot, ax_boxplot = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=categorical_feature, y=numeric_feature, data=df, ax=ax_boxplot)
            ax_boxplot.set_title(f'Box Plot: {numeric_feature} by {categorical_feature}')
            ax_boxplot.set_xlabel(categorical_feature)
            ax_boxplot.set_ylabel(numeric_feature)
            st.pyplot(fig_boxplot)
            st.markdown(
                "**Explanation:** A box plot compares the distribution of a numeric feature across different categories of a categorical feature. "
                "It shows the median, quartiles, and potential outliers for each category, allowing comparison of central tendencies and spread."
            )

        else:
            st.info("Bivariate analysis (Scatter Plot & Box Plot) is shown for Numeric vs Numeric and Numeric vs Categorical Feature combinations.")

    elif len(feature_2d_choices) == 1:
        st.warning("Please select two features for bivariate analysis.")
    elif len(feature_2d_choices) > 2:
        st.warning("Please select only two features for bivariate analysis.")

    st.divider()

    # --- Data Processing & Feature Engineering ---
    st.header("4. Data Processing and Feature Engineering")

    st.subheader("Numerical Feature Scaling (scale_numerical)")
    st.markdown(
        "Feature scaling is crucial for many machine learning algorithms. "
        "MoDeVa's `DataSet.scale_numerical` function supports various scaling methods. "
        "Select a numeric feature and a scaling method to see the effect on data distribution."
    )

    scaling_feature_choice = st.selectbox("Select Numeric Feature to Scale:", numeric_columns, key="scaling_feature_select")
    scaling_method = st.selectbox("Select Scaling Method:", ["standardize", "minmax", "log1p", "quantile"], index=1)

    if scaling_feature_choice and scaling_method:
        st.markdown(f"#### Scaled Distribution of {scaling_feature_choice} using {scaling_method}")

        scaled_df = df.copy() # Simulate scaling - In real MoDeVa, DataSet.scale_numerical would be used
        if scaling_method == "standardize":
            mean_val = scaled_df[scaling_feature_choice].mean()
            std_val = scaled_df[scaling_feature_choice].std()
            scaled_df[scaling_feature_choice] = (scaled_df[scaling_feature_choice] - mean_val) / std_val
            explanation = "**Standardize (StandardScaler):** Scales features to have zero mean and unit variance."
        elif scaling_method == "minmax":
            min_val = scaled_df[scaling_feature_choice].min()
            max_val = scaled_df[scaling_feature_choice].max()
            scaled_df[scaling_feature_choice] = (scaled_df[scaling_feature_choice] - min_val) / (max_val - min_val)
            explanation = "**MinMax (MinMaxScaler):** Scales features to a range between 0 and 1."
        elif scaling_method == "log1p":
            scaled_df[scaling_feature_choice] = np.log1p(scaled_df[scaling_feature_choice])
            explanation = "**Log1p (Log Transformation):** Applies a logarithmic transformation (log(1+x)) to handle skewed data and reduce variance."
        elif scaling_method == "quantile":
            quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42) # Simulate QuantileTransformer - from sklearn.preprocessing import QuantileTransformer
            scaled_df[scaling_feature_choice] = quantile_transformer.fit_transform(scaled_df[[scaling_feature_choice]]).flatten() # Fit and transform
            explanation = "**Quantile (QuantileTransformer):** Transforms features to a uniform or normal distribution. Useful for non-linear transformations and outlier robustness."
        else:
            explanation = ""

        fig_scaling, ax_scaling = plt.subplots(figsize=(10, 4))
        sns.histplot(scaled_df[scaling_feature_choice], kde=True, ax=ax_scaling)
        ax_scaling.set_title(f'Scaled Histogram and Density Plot of {scaling_feature_choice} ({scaling_method})')
        ax_scaling.set_xlabel(f'{scaling_feature_choice} (Scaled)')
        ax_scaling.set_ylabel('Frequency')
        st.pyplot(fig_scaling)
        st.markdown(f"**Explanation:** {explanation}")


    st.divider()

    st.subheader("Categorical Feature Encoding (encode_categorical)")
    st.markdown(
        "Categorical feature encoding converts categorical variables into numeric format. "
        "MoDeVa's `DataSet.encode_categorical` function supports one-hot and ordinal encoding. "
        "Select a categorical feature and an encoding method to see the transformed data."
    )

    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    if categorical_columns:
        encoding_feature_choice = st.selectbox("Select Categorical Feature to Encode:", categorical_columns, key="encoding_feature_select")
        encoding_method = st.selectbox("Select Encoding Method:", ["onehot", "ordinal"], index=0)

        if encoding_feature_choice and encoding_method:
            st.markdown(f"#### Encoded Data for {encoding_feature_choice} using {encoding_method}")

            encoded_df = df.copy() # Simulate encoding - In real MoDeVa, DataSet.encode_categorical would be used
            if encoding_method == "onehot":
                encoded_df = pd.get_dummies(encoded_df, columns=[encoding_feature_choice], prefix=encoding_feature_choice)
                explanation = "**One-Hot Encoding (OneHotEncoder):** Creates binary columns for each category, indicating presence or absence. Useful for nominal categorical features."
            elif encoding_method == "ordinal":
                unique_categories = encoded_df[encoding_feature_choice].unique()
                category_mapping = {category: i for i, category in enumerate(unique_categories)}
                encoded_df[encoding_feature_choice] = encoded_df[encoding_feature_choice].map(category_mapping)
                explanation = "**Ordinal Encoding (OrdinalEncoder):** Assigns a numerical order to categories. Suitable for ordinal categorical features where order matters."
            else:
                explanation = ""

            st.dataframe(encoded_df.head())
            st.markdown(f"**Explanation:** {explanation}")
    else:
        st.warning("No categorical columns available for encoding in the uploaded dataset.")


    st.divider()

    st.subheader("Numerical Feature Binning (bin_numerical)")
    st.markdown(
        "Numerical feature binning (or discretization) groups numeric values into bins or intervals. "
        "MoDeVa's `DataSet.bin_numerical` function supports uniform and quantile binning. "
        "Select a numeric feature and number of bins to see the binned data."
    )

    binning_feature_choice = st.selectbox("Select Numeric Feature to Bin:", numeric_columns, key="binning_feature_select")
    num_bins = st.slider("Number of Bins:", min_value=2, max_value=20, value=5)
    binning_method_choice = st.selectbox("Select Binning Method:", ["uniform", "quantile"], index=0)

    if binning_feature_choice and num_bins and binning_method_choice:
        st.markdown(f"#### Binned Feature {binning_feature_choice} using {binning_method_choice} method into {num_bins} bins")

        binned_df = df.copy() # Simulate binning - In real MoDeVa, DataSet.bin_numerical would be used
        if binning_method_choice == "uniform":
            binned_df[binning_feature_choice] = pd.cut(binned_df[binning_feature_choice], bins=num_bins, labels=False, include_lowest=True)
            explanation = "**Uniform Binning:** Divides the range of values into equal-width bins."
        elif binning_method_choice == "quantile":
            binned_df[binning_feature_choice] = pd.qcut(binned_df[binning_feature_choice], q=num_bins, labels=False, duplicates='drop') # Added duplicates='drop' to handle potential errors
            explanation = "**Quantile Binning:** Divides the range into bins with approximately equal numbers of data points."
        else:
            explanation = ""

        st.dataframe(binned_df.head())
        st.markdown(f"**Explanation:** {explanation} The original numeric feature is now transformed into discrete bins, represented by bin indices (starting from 0).")


st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")

from sklearn.preprocessing import QuantileTransformer #Import needed for Quantile scaling simulation
