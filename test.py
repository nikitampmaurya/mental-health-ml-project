import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
import seaborn as sns
import sklearn

module_versions = False # to note down version of each module, after checking the versions, set it false to avoid cluttering the terminal
if module_versions:
    print("Python version:", sys.version)
    print("Pandas:", pd.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Seaborn:", sns.__version__)
    print("Scikit-learn:", sklearn.__version__)

# Read in data
csv_path = Path(__file__).parent / "survey_data.csv"
df = pd.read_csv(csv_path)
pd.set_option('display.max_columns', None) # to view all the columns

show_data_info = True
if show_data_info:
    #print(df.head(),'\n') # view first few rows
    #print("No of rows and columns: ",df.shape, '\n') # 1259 rows and 27 columns
    #print(df.info(), '\n') # only age is the numeric column, others are str
    #print(df.isnull().sum(), '\n') # state, self_employed, work_interfere and comments columns have null values.
    # we will ignore the following comments: comments and timestamp (when they filled in the survey) because they are not useful. 

    df.columns = df.columns.str.lower() # converting all the columns into lowercase for easier handling

    #print(df['country'].value_counts()) # majorly US while other conuntry contribute by one or two entries. 
    # therefore we will drop country and state

    df = df.drop(['timestamp','country','state','comments'], axis = 1)

    columns_list = list(df.columns)

    for each_column in columns_list:
        print(each_column, df[each_column].unique(), '\n')

        # age has very low and high values, we address this shortly
        # gender has many 59 categories, we will have to simply it
        # self_employed has nan values which we will have to remove apart from yes and no.
        # family_history, treatment, remote_work, tech_company columns are alright with only two values (yes or no)
        # work_interfere also has nan values and ordinal categories (['Often', 'Rarely', 'Never', 'Sometimes', nan])
        # no_employees is also ordinal categories (has numeric range)
        # benefits, care_options, wellness_program, seek_help, anonymity,mental_vs_physcial are alright with three options yes, no, dont know/not sure
        # leave has ordinal categories ([     'Somewhat easy',         'Don't know', 'Somewhat difficult', 'Very difficult',          'Very easy'])
        # mental_health_consequence, phys_health_consequence, has three values only (['No', 'Maybe', 'Yes'])
        # coworkers and supervisor has three values (['Yes', 'No', 'Some of them'])
        # mental_health_interview, phys_health_interview has three values (['Maybe', 'No', 'Yes'])
        # obs_consequences is fine with two values yes or no

else: # when show_data_info is false, all the if statements will be skipped and only the else statements will run
    df.columns = df.columns.str.lower() # to keep column names lowercase for easier handling
    df = df.drop(['timestamp','country','state','comments'], axis = 1) # to drop irrelevant columns



