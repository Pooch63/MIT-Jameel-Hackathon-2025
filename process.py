import pandas as pd

# Suppress the darn SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

def process_classification(column: pd.Series, classificationMap: dict[str, int]) -> pd.Series:
    """
    Process a classification column to map string values to integers based on the provided mapping.
    
    Args:
        column (pd.Series): The column to process.
        classificationMap (dict[str, int]): A dictionary mapping string values to integers.
    
    Returns:
        pd.Series: The processed column with mapped integer values.
    """
    return column.map(classificationMap)
def process_flag(column: pd.Series) -> pd.Series:
    """
    Process the 'flag' column to convert 'yes' to 1 and 'no' to 0.
    """
    return process_classification(column, {'Y': 1, 'N': 0})
def process_gender(column: pd.Series) -> pd.Series:
    """
    Process the 'gender' column to convert 'F' to 1 and 'M' to 0.
    """
    return process_classification(column, {'F': 1, 'M': 0})
def process_race(column: pd.Series) -> pd.Series:
    """
    Process the 'race' column to convert races to integers.
    """
    return process_classification(column, {
        'White': 0,
        'Black': 1,
        'Asian/Pacific Islander': 2,
        'American Indian/Alaska Native': 3,
        'Unknown': 4,
        'Other': 5,
    })

state_codes = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
state_to_int = {code: i for i, code in enumerate(state_codes)}
def process_state(column: pd.Series) -> pd.Series:
    """
    Process the 'state' column to convert state codes to integers.
    """
    return column.map(state_to_int)
def process_education_level(column: pd.Series) -> pd.Series:
    """
    Process the 'education_level' column to convert education levels to integers.
    """
    return process_classification(column, {
        '': 0,
        'Some College': 1,
        'College Degree': 2,
        'Graduate Degree': 3,
        'High School or less': 4,
        'Trade School': 5,
    })
def process_allergy(column: pd.Series) -> pd.Series:
    """
    Process the 'allergy' column to convert allergy types to integers.
    """
    return process_classification(column, {
        'Other': 1,
        'TreeNut': 2,
        'Egg': 3,
        'Shellfish': 4,
        'Soybean': 5,
        'Fish': 6,
        'Peanut': 7,
    }).fillna(0)
def process_visit_type(column: pd.Series) -> pd.Series:
    """
    Process the 'visit_type' column to convert visit types to integers.
    """
    return process_classification(column, {
        'ER': 0,
        'Inpatient': 1,
        'Outpatient': 2,
    }).fillna(5)
def process_visit_level(column: pd.Series) -> pd.Series:
    """
    Process the 'visit_level' column to convert visit levels to integers.
    """
    return process_classification(column, {
        'Primary': 0,
        'Speciality': 1,
        'Other': 2
    }).fillna(2)

def date_to_timestamp(date: str) -> int:
    """
    Convert a date string in the format 'YYYY-MM-DD' to a timestamp.
    """
    return pd.Timestamp(date).timestamp()
def time_difference(start: str, end: str) -> int:
    """
    Calculate the difference in seconds between two date strings.
    """
    start_timestamp = date_to_timestamp(start)
    end_timestamp = date_to_timestamp(end)
    return int(end_timestamp - start_timestamp)

def analyze_dataframe(dataframe: pd.DataFrame, log=True):
    """
    Analyze the dataframe and print the number of rows and columns.
    """

    numbers = []
    flags = []
    objects = []
    for col in dataframe.columns.values:
        dtype = dataframe.dtypes[col]
        if dtype == 'int64' or dtype == 'float64':
            # If it ranges from 0 to 1, it's a flag
            if dataframe[col].min() == 0 and dataframe[col].max() == 1:
                flags.append((col, dtype))
            else:
                numbers.append((col, dtype))
        else:
            objects.append((col, dtype))
        
    # print(f"Numbers: {num_str}")
    # print(f"Objects: {object_str}")

    # Find 5-number summary for numerical columns
    if log:
        num_str = ""
        for col, dtype in numbers:
            num_str += f"{col}: {dtype}, "
        object_str = ""
        for col, dtype in objects:
            object_str += f"{col}: {dtype}, "
        for col in numbers:
            print(f"5-number summary for {col[0]}:\n{dataframe[col[0]].describe()}")

    return (numbers, flags, objects)

def process_dataframe(dataframe: pd.DataFrame, remove=True) -> pd.DataFrame:
    """
    Process the dataframe by converting columns to appropriate types, filling in missing values, and applying transformations.
    """

    # Remove unnecessary columns
    if remove:
        for v in range(1, 10 + 1):
            dataframe.drop(columns=[f'Visit{v}_Dx{d}' for d in range(1, 3 + 1)], inplace=True)
        for col in dataframe.columns:
            if 'Date' in col:
                dataframe.drop(columns=[col], inplace=True)
        dataframe.drop(columns=['Patient_ID'], inplace=True)
    
    # Hard-coded logic: Process sex, race, state, education level
    dataframe['Sex'] = process_gender(dataframe['Sex'])
    dataframe['Race'] = process_race(dataframe['Race'])
    dataframe['State'] = process_state(dataframe['State'])
    dataframe['Education_Level'] = process_education_level(dataframe['Education_Level'])
    dataframe['Childhood_Allergy_History'] = process_flag(dataframe['Childhood_Allergy_History'])
    dataframe['Allergen_Type'] = process_allergy(dataframe['Allergen_Type'])
    dataframe['Family_History'] = process_flag(dataframe['Family_History'])

    # Process visits
    for i in range(1, 10 + 1):
        dataframe[f'Visit{i}_Type'] = process_visit_type(dataframe[f'Visit{i}_Type'])
        dataframe[f'Visit{i}_Level'] = process_visit_level(dataframe[f'Visit{i}_Level'])
        # Fill severity with 0 if NaN
        dataframe[f'Visit{i}_ER_Severity'] = dataframe[f'Visit{i}_ER_Severity'].fillna(0)
    for i in range(1, 5 + 1):
        dataframe[f'Lab{i}_Flag'] = process_classification(dataframe[f'Lab{i}_Flag'], {
            'Normal': 0,
            'Abnormal': 1,
        }).fillna(0)
        dataframe[f'Lab{i}_Test'] = process_classification(dataframe[f'Lab{i}_Test'], {
            'Other': 0,
            'Cholestrol': 1,
            'OralFoodchallenge': 2,
            'CBC': 3,
            'SpecificIgE': 4,
            'Glucose': 5,
            'TotalIgE': 6,
            'SkinPrickTest': 7,
            'Eosinophils': 8,
        }).fillna(0)

    print("Done with processing classifications")

    # For every person, when visits stop, fill them with last heights and weights
    last_widths = dataframe['Visit1_Weight_lb'].copy()
    last_heights = dataframe['Visit1_Height_in'].copy()
    last_ages = dataframe['Visit1_Age'].copy()
    for i in range(2, 10 + 1):
        weight_col = dataframe[f'Visit{i}_Weight_lb']
        height_col = dataframe[f'Visit{i}_Height_in']
        ages_col = dataframe[f'Visit{i}_Age']
        
        for j, row in dataframe.iterrows():
            if pd.isnull(weight_col[j]):
                weight_col[j] = last_widths[j]
            else:
                last_widths[j] = weight_col[j]
                
            if pd.isnull(height_col[j]):
                height_col[j] = last_heights[j]
            else:
                last_heights[j] = height_col[j]

            if pd.isnull(ages_col[j]):
                ages_col[j] = last_ages[j]
            else:
                last_ages[j] = ages_col[j]

        print(f"{((i - 1) * 100 / 9):<2}% done with filling heights and weights")

    numbers, flags, objects = analyze_dataframe(dataframe, log=False)
    # Fill in missing values for numerical columns with the mean
    for num in numbers:
        col = num[0]
        if dataframe[col].isnull().any():
            mean_value = dataframe[col].mean()
            dataframe[col].fillna(mean_value)
    for flag in flags:
        col = flag[0]
        if dataframe[col].isnull().any():
            dataframe[col].fillna(0)
