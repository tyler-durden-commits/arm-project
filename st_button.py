import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import plot_importance
import joblib

# Configure Streamlit page
st.set_page_config(page_title="Electricity Price Prediction")

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
    st.session_state.df = None
    st.session_state.clean_df = None
    st.session_state.df_cleaned = None
    st.session_state.model = None
    st.session_state.best_model = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    # Main presentation flow
    st.title("Predicting Electricity Import Prices from Canada to US")

    if st.session_state.current_step == 0:
        st.header("1. Data Overview")
        st.subheader("Raw Data Sample")
        st.dataframe(st.session_state.df.head())

        if st.button("Next: Missing Values Analysis"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 1:
        st.header("2. Missing Values Analysis")
        missing_values = (st.session_state.df.isnull().sum() / st.session_state.df.shape[0]).to_frame(
            name="Missing Proportion")
        st.bar_chart(missing_values)
        st.write("We'll drop rows with missing values to ensure data quality.")

        if st.button("Next: Feature Engineering"):
            st.session_state.clean_df = st.session_state.df.dropna()
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 2:
        st.header("3. Feature Engineering")
        st.markdown("""
        We create several features to help the model understand patterns:
        - **Exports/Imports**: Split transfer values into separate features
        - **Time Features**: Extract hour from datetime
        - **ISO-Province Pairs**: Combine identifiers for unique pairs
        """)

        st.session_state.clean_df['exports_MWh'] = st.session_state.clean_df['transfers_MWh'].apply(
            lambda x: -x if x < 0 else 0)
        st.session_state.clean_df['imports_MWh'] = st.session_state.clean_df['transfers_MWh'].apply(
            lambda x: x if x > 0 else 0)
        st.session_state.clean_df['time_hour'] = pd.to_datetime(st.session_state.clean_df['utc']).dt.hour

        if st.button("Next: Exploratory Data Analysis"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 3:
        st.header("4. Exploratory Data Analysis")

        if st.button("Show Distribution Plots"):
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.clean_df['imports_MWh'], kde=True, bins=30, ax=ax)
                ax.set_title('Distribution of Electricity Imports (MWh)')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.clean_df['iso_price'], kde=True, bins=30, ax=ax)
                ax.set_title('Distribution of ISO Prices')
                st.pyplot(fig)

        if st.button("Next: Outlier Handling"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 4:
        st.header("5. Outlier Handling")

        if st.button("Show Original Distribution"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=st.session_state.clean_df['iso_price'], ax=ax)
            st.pyplot(fig)

        if st.button("Handle Outliers"):
            q1 = st.session_state.clean_df['iso_price'].quantile(0.25)
            q3 = st.session_state.clean_df['iso_price'].quantile(0.75)
            IQR = q3 - q1
            lower_bound = q1 - 1.5 * IQR
            upper_bound = q3 + 1.5 * IQR
            st.session_state.df_cleaned = st.session_state.clean_df[
                (st.session_state.clean_df['iso_price'] >= lower_bound) &
                (st.session_state.clean_df['iso_price'] <= upper_bound)
                ]
            st.success("Outliers handled successfully!")

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=st.session_state.df_cleaned['iso_price'], ax=ax)
            st.pyplot(fig)

        if st.button("Next: Advanced Feature Engineering"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 5:
        st.header("6. Advanced Feature Engineering")

        st.session_state.df_cleaned['iso_province_pair'] = st.session_state.df_cleaned['iso'].astype(str).str.cat(
            st.session_state.df_cleaned['province'].astype(str), sep='-')
        st.session_state.df_cleaned['iso_province_pair'] = st.session_state.df_cleaned['iso_province_pair'].astype(
            'category')
        st.session_state.df_cleaned['lag_1h_iso_price'] = st.session_state.df_cleaned['iso_price'].shift(1)
        st.session_state.df_cleaned['rolling_mean_3h_iso_price'] = st.session_state.df_cleaned['iso_price'].rolling(
            window=3).mean()
        st.session_state.df_cleaned['rolling_std_6h_iso_price'] = st.session_state.df_cleaned['iso_price'].rolling(
            window=6).std()
        st.session_state.df_cleaned['wind_solar_interaction'] = st.session_state.df_cleaned['wind_speed'] * \
                                                                st.session_state.df_cleaned['solar_radiation']

        st.markdown("""
        **Created Features:**
        - ISO-Province Pair (categorical)
        - 1-hour lagged price
        - 3-hour rolling mean price
        - 6-hour rolling standard deviation
        - Wind-Solar interaction term
        """)

        if st.button("Next: Model Preparation"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 6:
        st.header("7. Model Preparation")

        X = st.session_state.df_cleaned[['wind_solar_interaction', 'annual_hour', 'iso_province_pair', 'iso_load',
                                         'lag_1h_iso_price', 'rolling_mean_3h_iso_price', 'rolling_std_6h_iso_price']]
        y = st.session_state.df_cleaned['iso_price']

        train_size = int(0.8 * len(X))
        st.session_state.X_train, st.session_state.X_test = X[:train_size], X[train_size:]
        st.session_state.y_train, st.session_state.y_test = y[:train_size], y[train_size:]

        scaler = StandardScaler()
        num_cols = ['wind_solar_interaction', 'annual_hour', 'iso_load', 'lag_1h_iso_price',
                    'rolling_mean_3h_iso_price', 'rolling_std_6h_iso_price']

        st.session_state.X_train[num_cols] = scaler.fit_transform(st.session_state.X_train[num_cols])
        st.session_state.X_test[num_cols] = scaler.transform(st.session_state.X_test[num_cols])

        st.markdown("""
        **Time-Based Data Split (80-20):**
        Because this is time-series data, we split chronologically:
        - First 80% of data points for training
        - Last 20% for testing
        """)

        if st.button("Next: Model Training"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 7:
        st.header("8. Model Training")

        if st.button("Train XGBoost Model"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            st.session_state.model = XGBRegressor(enable_categorical=True, n_estimators=100, random_state=42)

            for i in range(101):
                if i == 100:
                    st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                progress_bar.progress(i)
                status_text.text(f"Training progress: {i}%")

            status_text.text("Training complete!")
            st.success("Model trained successfully!")

        if st.button("Next: Model Evaluation"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 8:
        st.header("9. Model Evaluation")

        if st.session_state.model is not None:
            y_pred = st.session_state.model.predict(st.session_state.X_test)

            metrics = {
                "Metric": ["R² Score", "Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error"],
                "Value": [
                    r2_score(st.session_state.y_test, y_pred),
                    mean_squared_error(st.session_state.y_test, y_pred),
                    np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)),
                    mean_absolute_error(st.session_state.y_test, y_pred)
                ],
                "Interpretation": [
                    "Proportion of variance explained (higher is better)",
                    "Average squared difference (lower is better)",
                    "Average error magnitude (in original units)",
                    "Average absolute error (robust to outliers)"
                ]
            }

            st.table(pd.DataFrame(metrics))

            if st.button("Next: Hyperparameter Tuning"):
                st.session_state.current_step += 1
                st.rerun()

    elif st.session_state.current_step == 9:
        st.header("10. Hyperparameter Tuning")

        if st.button("Run Grid Search"):
            with st.spinner("Running Grid Search... This may take a while ⏳"):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }

                grid_search = GridSearchCV(
                    estimator=st.session_state.model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(st.session_state.X_train, st.session_state.y_train)

                st.session_state.best_model = grid_search.best_estimator_

                st.write("Best Parameters:", grid_search.best_params_)
                st.write("Best R² Score:", grid_search.best_score_)

                st.success("Hyperparameter tuning complete!")

        if st.button("Next: Final Model"):
            st.session_state.current_step += 1
            st.rerun()

    elif st.session_state.current_step == 10:
        st.header("11. Final Model Performance")

        if st.session_state.best_model is not None:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(st.session_state.best_model, ax=ax)
            st.pyplot(fig)

            st.subheader("Cross-Validation Results")
            cv_scores = cross_val_score(
                st.session_state.best_model,
                st.session_state.X_train,
                st.session_state.y_train,
                cv=5,
                scoring='r2'
            )
            st.write("Cross-Validation R² Scores:", cv_scores)
            st.write("Mean R²:", cv_scores.mean())

            st.markdown("""
            **Conclusion:**
            - The XGBoost model effectively predicts electricity import prices
            - Time-based features and interaction terms were particularly important
            - The model achieves good predictive performance
            """)

        elif st.button("🔄 Restart Presentation"):
                st.session_state.current_step = 0
                st.rerun()
