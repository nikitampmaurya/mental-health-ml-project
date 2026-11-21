import sys
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter


module_versions = False # to note down version of each module, after checking the versions, set it false to avoid cluttering the terminal
if module_versions:
    print("Python version:", sys.version)
    print("Pandas:", pd.__version__)
    print("NumPy:", np.__version__)
    print("Matplotlib,pyplot:", plt.matplotlib.__version__)
    print("Seaborn:", sns.__version__)
    print("Scikit-learn:", sklearn.__version__)

# Read in data
csv_path = Path(__file__).parent / "survey_data.csv"
df = pd.read_csv(csv_path)
pd.set_option('display.max_columns', None) # to view all the columns

show_data_info = False
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

# cleaning gender column 
df['gender'] = df['gender'].replace(['Female', 'female', 'Cis Female', 'F', 'Woman', 'f', 'Femake', 'woman', 'cis-female/femme', 'femail', 'Female (cis)', 'Female '], 'Female') 
df['gender'] = df['gender'].replace(['M', 'Male', 'male', 'm', 'Male-ish', 'maile', 'Cis Male', 'Mal', 'Male (CIS)', 'Make', 'Cis Man', 'Malr', 'cis male', 'Mail',  'msle', 'Man' ,  'Male '], 'Male') 
df['gender'] = df['gender'].replace(['Trans-female','something kinda male?', 'queer/she/they','non-binary',  'Nah','Enby','fluid','Genderqueer',  'Androgyne', 'Agender',  'queer', 'Female (trans)',  'Neuter', 'Trans woman', 'male leaning androgynous', 'Guy (-ish) ^_^' , 'ostensibly male, unsure what that really means'], 'Others')


if show_data_info:
    print("total types of genders (After Cleaning): ", df["gender"].nunique()) # now only three types
    print(df['gender'].value_counts()) # majorly men (986) followed by female (246) and then others (18), very high imbalance between genders
                                         

#### Exploratory Data Analysis ####

Run_EDA = True # to run code under if statement and we can switch this off by changing it to False

# let us split data into 80% train and 20% test. So we dont perform EDA on test since it leads to data leakage/studying for exam by using actual exam paper (cheating on exam)
train, test = train_test_split(df, test_size = 0.2, stratify = df['treatment'], random_state=1) # random_state is used to make your random train-test split reproducible

colour_palette = {"Yes":"#0072B2","No":"#D55E00"} # setting colour for treatment columns

if Run_EDA:
    print('Dimensions of Train Dataset: ', train.shape) # 1000 rows and 23 columns
    print('Diminsions of Test Dataset: ', test.shape) # 250 rows and 23 columns

    # Starting with our target variable: Treatment column 

    sns.set_style("white") # setting the background of plot 
    plt.figure(figsize=(10, 8)) # setting dimensions of plot
    treatment_proportion = round(train['treatment'].value_counts(normalize = True).reset_index(name = 'percentage'),3) # count total no of yes and no within treatment column and turn it into proportions, then put into dataframe with two columns (treatment, percentage)
    ax = sns.barplot(x = 'treatment', y = 'percentage', data = treatment_proportion, palette = colour_palette) # create bar plot with these columns and colour

    for bar in ax.patches: # for each bar 
        width = bar.get_width() # get width to place label
        height = bar.get_height() # get height to place label
        x,y = bar.get_xy() # get x and y or-ordinate on left hand side bottom
        ax.annotate(f'{height:.1%}', (x + bar.get_width() / 2,height + 0.005), fontsize=16) # turn proportion into percent, place in the calculated center


    ax.set_xlabel("Sought treatment", fontsize = 16) # rename the x axis label
    ax.set_ylabel("Percentage", fontsize = 16) # rename the y axis label
    ax.set_xticklabels(['Yes', 'No'], fontsize=16, fontweight='bold') # make yes and no bold
    ax.tick_params(axis='y', labelsize=16) # make tick bold
    ax.yaxis.set_major_formatter(PercentFormatter(1)) # turn proportions into percent
    plt.title('Treatment percentage', fontsize = 16, fontweight = "bold") # make title bold 
    plt.show() # to show the plot

    # My observations: 
    # # Result: Nearly equal proportions of people sought (50.5%) and did not seek treatment (49.5%).
    # Both classes are well represented, so resampling is not required to address class imbalance.
    # When evaluating the model, consider recall to minimise false negatives.


    # Next is Age Column

    sns.set_style("white") # setting the background of plot 
    plt.figure(figsize=(10, 8)) # setting dimensions of plot
    ax = sns.histplot(x="age", data = train, bins = 20, hue="treatment", palette=colour_palette) # create histogram to view distributions with these columns and colour

    ax.set_xlabel("Age (in years)", fontsize = 16) # rename the x axis label
    plt.title("Age of participants", fontsize = 16, fontweight="bold") # set title, its font style and size
    plt.show()
    # distribution between two group is very overlapping, may not alone stand as strong predictor


