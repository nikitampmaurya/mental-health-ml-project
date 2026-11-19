import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
    print(df.head(),'\n') # view first few rows
    print("No of rows and columns: ",df.shape, '\n') # 1259 rows and 27 columns
    print(df.info(), '\n') # only age is the numeric column, others are str
    print(df.isnull().sum(), '\n') # state, self_employed, work_interfere and comments columns have null values.
    # we will ignore the following columns: comments and timestamp (when they filled in the survey) because they are not useful. 

    df.columns = df.columns.str.lower() # converting all the column names into lowercase for easier handling

    print(df['country'].value_counts()) # majorly US while other country contribute by one or two entries. 
    # therefore we will drop country and state

    df = df.drop(['timestamp','country','state','comments'], axis = 1) # removing irrelevant columns

    columns_list = list(df.columns) # list of columns

    for each_column in columns_list:
        print(each_column, df[each_column].unique(), '\n') # print unique values of each column

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


#### Data Cleaning ####

if show_data_info:
    # First checking for NA value
    print("No of na values: ",df["age"].isna().sum(), "\n") # not found

    df_sorted = df.sort_values(by='age', ascending = True) # sorting to understand highest and lowest values
    print("Young Age: ",df_sorted["age"].head(10),"\n") # top 10 lowest values of age
    print("Old Age: ",df_sorted["age"].tail(10),"\n") # top 10 highest value of age

    plt.figure() # create blank plot
    plt.hist(df['age'], bins = 10, edgecolor = "black") # draw histogram by dividing age into 10 intervals
    plt.title('Age (Before Cleaning)') # one bin because because the maximum value is 999999, Matplotlib creates very wide bins. 
    # Most normal ages get grouped into the first bin, making the histogram look as though there is only one peak near zero.

df = df[(df["age"] >= 18) & (df["age"] <= 68)] # keeping only adults and senior candidates only uptil 68.

if show_data_info: 
    
    plt.figure() 
    plt.hist(df['age'], bins=10, edgecolor='black')
    plt.title('Ages (After cleaning') # majority of the candidates are between 20 and 50 of age.

    # Show plots
    plt.show()

if show_data_info:
    print("Total types of genders (Before cleaning): ", df["gender"].nunique()) # we have 46 genders
    # they are as follows:
    # ['Female', 'M', 'Male','male','female','m', 'Male-ish', 'maile', 'Trans-female', 'Cis Female', 'F', 'something kinda male?', 'Cis Male',
    # 'Woman', 'f', 'Mal', 'Male (CIS)', 'queer/she/they', 'non-binary', 'Femake', 'woman', 'Make', 'Nah', 'Enby', 'fluid', 'Genderqueer', 'Female ',
    # 'Androgyne', 'Agender', 'cis-female/femme', 'Guy (-ish) ^_^', 'male leaning androgynous','Male ', 'Man', 'Trans woman', 'msle', 'Neuter',
    # 'Female (trans)', 'queer', 'Female (cis)', 'Mail', 'cis male', 'Malr', 'femail','Cis Man', 'ostensibly male, unsure what that really means']


df['gender'] = df['gender'].replace(['Female', 'female', 'Cis Female', 'F', 'Woman', 'f', 'Femake', 'woman', 'cis-female/femme', 'femail', 'Female (cis)', 'Female '], 'Female')
df['gender'] = df['gender'].replace(['M', 'Male', 'male', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Cis Man', 'Malr', 'cis male', 'Mail',  'msle', 'Man' ,  'Male '], 'Male')
df['gender'] = df['gender'].replace(['Trans-female','something kinda male?', 'queer/she/they','non-binary',  'Nah','Enby','fluid','Genderqueer',  'Androgyne', 'Agender',  'queer', 'Female (trans)',  'Neuter', 'Trans woman', 'male leaning androgynous', 'Guy (-ish) ^_^' , 'ostensibly male, unsure what that really means'], 'Others')


if show_data_info:
    print("total types of genders (After Cleaning): ", df["gender"].nunique()) # now only three types
    print(df['gender'].value_counts()) # majorly men (986) followed by female (246) and then others (18), very high imbalance between genders
                                         
                                   
                                           
                                      
                                           




