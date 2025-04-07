import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("Lab Result Predictor Dashboard")
st.write("""
Upload your dataset to get insights and understand the impact of new data columns on predictions. 
The app will preprocess your data, check for missing columns, and provide visualizations and insights.
""")

# Sidebar for filtering
st.sidebar.header("Filter Data")

# File Uploader (allow both CSV and Excel files)
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Check file type and read accordingly
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        data = pd.read_excel(uploaded_file)

    st.write("### Original Dataset", data.head())

    # Required Columns
    required_columns = [
        "company_name", "location", "gender", "age_in_years", "collection_date_and_time", "type_of_testing",
        "declared_medication", "alcohol_result", "breath_test_result_1", "breath_test_result_2", "company_cut_off_level",
        "drug_result", "drugs_non_negative", "medical_history", "recent_symptoms", "smoking_status", "job_role",
        "work_shift", "hours_worked", "safety_training", "incident_history", "reason_for_testing", "recent_accident",
        "substance_levels", "education_level", "stress_level", "lab_result"
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.warning(f"The following required columns are missing: {', '.join(missing_columns)}. Please ensure your dataset includes these columns.")
    else:
        st.success("All required columns are present!")

    # Preprocessing (fill missing values)
    data.fillna("Unknown", inplace=True)

    # Convert date column to datetime and extract features
    data['collection_date_and_time'] = pd.to_datetime(data['collection_date_and_time'])
    data['year'] = data['collection_date_and_time'].dt.year
    data['month'] = data['collection_date_and_time'].dt.month
    data['day_of_week'] = data['collection_date_and_time'].dt.dayofweek
    data['weekOfYear'] = data['collection_date_and_time'].dt.isocalendar().week
    data['hour'] = data['collection_date_and_time'].dt.hour
    data['minutes'] = data['collection_date_and_time'].dt.minute
    data['seconds'] = data['collection_date_and_time'].dt.second

    # Drop original 'collection_date_and_time' column
    data = data.drop(columns=['collection_date_and_time'])

    st.write("### Preprocessed Dataset", data.head())

    # Sidebar filters for specific columns
    st.sidebar.header("Select Filters")

    # Select filters for additional columns
    gender_filter = st.sidebar.multiselect("Select Gender", options=["All"] + list(data["gender"].unique()))
    age_filter = st.sidebar.slider("Select Age Range", min_value=int(data["age_in_years"].min()), 
                                   max_value=int(data["age_in_years"].max()), 
                                   value=(int(data["age_in_years"].min()), int(data["age_in_years"].max())))

    # More filters for new columns
    company_filter = st.sidebar.multiselect("Select Company", options=["All"] + list(data["company_name"].unique()))
    location_filter = st.sidebar.multiselect("Select Location", options=["All"] + list(data["location"].unique()))
    type_of_testing_filter = st.sidebar.multiselect("Select Type of Testing", options=["All"] + list(data["type_of_testing"].unique()))
    declared_medication_filter = st.sidebar.multiselect("Select Declared Medication", options=["All"] + list(data["declared_medication"].unique()))
    alcohol_result_filter = st.sidebar.multiselect("Select Alcohol Test Result", options=["All"] + list(data["alcohol_result"].unique()))
    breath_test_result_filter = st.sidebar.multiselect("Select Breath Test Result", options=["All"] + list(data["breath_test_result_1"].unique()))
    drug_result_filter = st.sidebar.multiselect("Select Drug Test Result", options=["All"] + list(data["drug_result"].unique()))
    drugs_non_negative_filter = st.sidebar.multiselect("Select Drugs Non-Negative", options=["All"] + list(data["drugs_non_negative"].unique()))

    # Apply filters to data
    filtered_data = data.copy()

    # Apply all filters to data
    if "All" not in gender_filter:
        filtered_data = filtered_data[filtered_data["gender"].isin(gender_filter)]
    filtered_data = filtered_data[filtered_data["age_in_years"].between(age_filter[0], age_filter[1])]
    if "All" not in company_filter:
        filtered_data = filtered_data[filtered_data["company_name"].isin(company_filter)]
    if "All" not in location_filter:
        filtered_data = filtered_data[filtered_data["location"].isin(location_filter)]
    if "All" not in type_of_testing_filter:
        filtered_data = filtered_data[filtered_data["type_of_testing"].isin(type_of_testing_filter)]
    if "All" not in declared_medication_filter:
        filtered_data = filtered_data[filtered_data["declared_medication"].isin(declared_medication_filter)]
    if "All" not in alcohol_result_filter:
        filtered_data = filtered_data[filtered_data["alcohol_result"].isin(alcohol_result_filter)]
    if "All" not in breath_test_result_filter:
        filtered_data = filtered_data[filtered_data["breath_test_result_1"].isin(breath_test_result_filter)]
    if "All" not in drug_result_filter:
        filtered_data = filtered_data[filtered_data["drug_result"].isin(drug_result_filter)]
    if "All" not in drugs_non_negative_filter:
        filtered_data = filtered_data[filtered_data["drugs_non_negative"].isin(drugs_non_negative_filter)]

    st.write(f"### Filtered Dataset")
    st.write(filtered_data.head())

    # Descriptive Statistics
    st.subheader("Dataset Statistics")
    st.write(filtered_data.describe(include="all"))

    # Correlation Matrix
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_cols.corr()

    if len(correlation) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        st.pyplot(plt)
    else:
        st.warning("Not enough numerical columns for a correlation heatmap.")

    # Insights and Analytics

    # Drug Result by Company
    st.subheader("Drug Result by Company")
    drug_result_by_company = filtered_data.groupby('company_name')['drug_result'].value_counts().unstack().fillna(0)
    st.bar_chart(drug_result_by_company)

    # Correlation between Work-Related Factors
    st.subheader("Correlation between Work-Related Factors (Hours Worked, Job Role, etc.)")
    work_related_cols = ['hours_worked', 'job_role', 'work_shift', 'safety_training']
    work_data = filtered_data[work_related_cols].apply(pd.to_numeric, errors='coerce')
    work_correlation = work_data.corr()
    if len(work_correlation) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(work_correlation, annot=True, cmap='coolwarm')
        plt.title("Work-Related Factors Correlation")
        st.pyplot(plt)

    # Stress Level vs Substance Levels Analysis
    st.subheader("Stress Level vs Substance Levels Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="stress_level", y="substance_levels", data=filtered_data, ax=ax)
    ax.set_title("Stress Level vs Substance Levels")
    st.pyplot(fig)

    # Class Distribution for Declared Medication Use
    st.subheader("Class Distribution for Declared Medication Use")
    declared_medication_dist = filtered_data['declared_medication'].value_counts()
    st.bar_chart(declared_medication_dist)

    # Job Role and Safety Training Impact on Test Results
    st.subheader("Job Role and Safety Training Impact on Test Results")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="job_role", hue="safety_training", data=filtered_data, ax=ax)
    ax.set_title("Job Role and Safety Training Impact on Test Results")
    st.pyplot(fig)

    # Incident History and Recent Accidents Correlation
    st.subheader("Incident History and Recent Accidents Correlation")
    incident_data = filtered_data[['incident_history', 'recent_accident']].apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(8, 6))
    sns.heatmap(incident_data.corr(), annot=True, cmap='coolwarm')
    plt.title("Incident History and Recent Accidents Correlation")
    st.pyplot(plt)

    st.subheader("Insights")

    st.write("""
    - The filters you apply will update the dataset and visualizations.
    - Explore the relationship between various features like work hours, stress levels, and test results.
    - Visualize correlations, distributions, and class distributions based on your selections.
    """)
else:
    st.info("Please upload a CSV or Excel file to proceed.")
