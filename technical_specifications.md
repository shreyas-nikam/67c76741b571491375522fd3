# MoDeVa Data Insights Explorer

## Overview

MoDeVa Data Insights Explorer is a single-page Streamlit application designed to introduce users to various dataset operations and visualizations available in MoDeVa. The app focuses on providing users with a high-level overview of essential data manipulation tasks, offering both practical examples and real-time visualizations.

## Technical Specifications

### Application Structure

The application is a Streamlit-based web application with a single-page layout containing several interactive sections. The main components are designed to facilitate ease of use and streamline the exploration process while enabling seamless user interactions.

### Dataset Details

- **Source**: The application employs a synthetic dataset designed to mimic the structure and characteristics of real-world data. This allows for a controlled demonstration of dataset operations and ensures reproducibility.
- **Content**: The dataset includes a diverse array of features, such as numeric values, categorical variables, and time-series data. This diversity ensures that all data operations and visualizations showcased can be applied to real-life scenarios.

### Key Features and Functionalities

#### 1. Basic Operations

- **Data Loading**: Users can load the synthetic dataset or upload their own datasets for analysis.
- **Summary Statistics**: Display a summary of the dataset using the `DataSet.summary` function. This includes key metrics such as count, mean, standard deviation, min, max, and quartiles.
- **Data Manipulation**: Perform basic data manipulation tasks to showcase how data can be transformed using MoDeVa's capabilities.

#### 2. Exploratory Data Analysis (EDA)

- **Interactive Charts**:
  - **Univariate Analysis**: Utilize `DataSet.eda_1d` to display histograms and density plots for individual numeric features.
  - **Bivariate Analysis**: Implement scatter plots, heatmaps, and boxplots using `DataSet.eda_2d` to explore relationships between pairs of features.
- **Annotations & Tooltips**: Provide detailed insights and explanations directly on the charts to aid data interpretation.

#### 3. Data Processing and Feature Engineering

- **Scaling**: Implement sliders and dropdowns to scale numeric features using `DataSet.scale_numerical`, showing real-time changes in data distribution.
- **Encoding**: Use widgets to apply and visualize categorical encoding via `DataSet.encode_categorical`.
- **Binning**: Allow users to apply numerical binning using `DataSet.bin_numerical` and immediately view the results on the visualizations.

### User Interaction

- **Dynamic Visualizations**: Real-time updates for visualizations based on user-selected parameters, datasets, or features.
- **Input Forms and Widgets**: Users can interactively choose data attributes for analysis and visualization, observing immediate feedback through the UI.
- **Documentation and Help**: Built-in inline help text and tooltips guide users through each operation, explaining both functionality and relevance.

### Relation to Document

The application implements practical examples detailed in the "Basic Data Operations" and "Data Processing" sections of the referenced document, showcasing each method's capabilities through dynamic visual interactions. The specified Visualization Details are incorporated through interactive charts and annotations, enhancing user comprehension and allowing for direct engagement with analytical outcomes.

### References

- MoDeVa Documentation: Refer to the "Basic Data Operations" and "Data Processing" sections, which provide theoretical underpinnings for the implemented features.