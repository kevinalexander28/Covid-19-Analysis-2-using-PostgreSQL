''' 
Import Libraries =============================================================
'''
# operating system
import os
# data processing and manipulation
import numpy as np
import pandas as pd
# connect python to postgre
import psycopg2
import psycopg2.extras as extras
from psycopg2 import Error
from configparser import ConfigParser
from io import StringIO
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# statistics
from scipy import stats
# machine learning
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# path graphviz
os.environ["PATH"] += os.pathsep + 'C:\\Users\\kevin\\anaconda3\\Library\\bin\\graphviz'


# POSTGRESQL COMMAND
'''
-> TO CHECK EXISTING TABLES
postgres=# \dt
'''

# FUNCTIONS
''' 
GET CONNECTION PARAMETERS ====================================================
'''
# get connection parameters from file 'database.ini'
def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'
                        .format(section, filename))
    return db

''' 
CHECK CONNECTION =============================================================
'''
# connect to the PostgreSQL database server
def is_connect():
    conn = None
    try:
        # read connection parameters using function 'config'
        params = config()
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create a cursor
        cur = conn.cursor()
        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
        print("Successfully Connected!")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return 1
    finally:
        # close the communication with the PostgreSQL
        if conn is not None:
            conn.close()
            print('Database connection closed.')

''' 
EXECUTE QUERY ================================================================
'''
# database operations in the PostgreSQL database
# for create new table or drop existing table or more
def execute_query(query):
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        # create a cursor to perform database operations
        cur = conn.cursor()
        # executing a SQL query for create table
        cur.execute(query)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return 1
    print("Execution Succeed!")
        
''' 
INSERT DATA TO TABLE =========================================================
'''
# using psycopg2.extras.execute_values() to insert the dataframe
def inset_table_execute_values(df, table):
    # try open connection
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return 1
    # create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("Insert Data Succeed")
    cursor.close()

''' 
SELECT DATA FROM TABLE =======================================================
'''
# database operations in the PostgreSQL database
# for get data from database
def get_data_from_database(query):
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        # create a cursor to perform database operations
        cur = conn.cursor()
        # executing a SQL query for create table
        cur.execute(query)
        # get column names
        column_names = [row[0] for row in cur.description]
        # fetch all data from table
        record = cur.fetchall()
        # save to dataframe
        df = pd.DataFrame(record, columns=column_names)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()    
        return df
    
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return 1
    print("Select Data Succeed")
            
            
if __name__ == "__main__":
    ''' 
    CHECKING CONNECTION TO DATABASE ==========================================
    '''
    # Test Connection
    is_connect()
    
    ''' 
    PREPARE DATASETS =========================================================
    '''
    # main dataset
    df_covid = pd.read_csv('data/covid_19_data.csv',sep=';')
    print(df_covid)
    # region and climate dataset
    df_covid_cat = pd.read_csv('data/covid_19_data_cat.csv')
    print(df_covid_cat)
    
    # convert data type
    df_covid['date'] = df_covid['date'].astype('datetime64[ns]')
    df_covid['date'] = pd.to_datetime(df_covid['date'], format='%y%m%d').dt.date
    print(df_covid)
    
    ''' 
    DATABASE OPERATIONS ======================================================
    '''
    # drop if exists
    query = "DROP TABLE COVID"
    execute_query(query)
    
    query = "DROP TABLE COVID_CAT"
    execute_query(query)
    
    # create table covid
    query = """
            CREATE TABLE COVID (
                DATE DATE,
                PROVINCE VARCHAR(50),
                COUNTRY VARCHAR(50),
                CONFIRMED INT,
                DEATHS INT,
                RECOVERED INT
            )
            """
    # execute query
    execute_query(query)
    
    # create table covid_cat
    query = """
            CREATE TABLE COVID_CAT (
                COUNTRY VARCHAR(50),
                REGION VARCHAR(50),
                CLIMATE VARCHAR(50)
            )
            """
    # execute query
    execute_query(query)
    
    # insert data from dataframe covid to table 'covid'
    inset_table_execute_values(df_covid, 'COVID')
    
    # insert data from dataframe covid_cat to table 'covid_cat'
    inset_table_execute_values(df_covid_cat, 'COVID_CAT')
    
    # check that the values were indeed inserted
    execute_query("SELECT COUNT(*) FROM COVID;")
    # execute_query("SELECT COUNT(*) FROM COVID_CAT;")
    
    # clear the table if needed
    # execute_query("DELETE FROM COVID WHERE TRUE;")
    # execute_query("DELETE FROM COVID_CAT WHERE TRUE;")
    
    # get data from table 'covid'
    query = "SELECT * FROM COVID"
    # save to dataframe 'df'
    df = get_data_from_database(query)
    print(df_covid)
    
    # get data from table 'covid_cat'
    query = "SELECT * FROM COVID_CAT"
    # save to dataframe 'df_cat'
    df_cat = get_data_from_database(query)
    print(df_covid_cat)
    
    ''' 
    DATASET INFORMATION ======================================================
    '''
    print("DATASET INFORMATION:")
    # No. of columns and rows
    print("Number of Columns:", len(df.columns))
    print("Number of Rows:", len(df))
    
    # No. of countries
    print("Number of Countries:", len(df.country.unique()))
    
    # Convert data type
    df['date'] = df['date'].astype('datetime64[ns]')
    df['date'] = pd.to_datetime(df['date'], format='%y%m%d').dt.date
    
    # No. of days
    print("Number of Days:", len(df.date.unique()))
    print("Since:", min(df.date))
    print("Until:", max(df.date))
    
    ''' 
    DATA PREPROCESSING =======================================================
    '''
    # replace with this data
    replace_country = {"('St. Martin',)":'St. Martin',
                       ' Azerbaijan':'Azerbaijan',
                       'Cabo Verde':'Cape Verde',
                       'Congo (Brazzaville)':'Congo',
                       'Congo (Kinshasa)':'Congo',
                       'North Ireland':'Ireland',
                       'North Macedonia':'Macedonia',
                       'occupied Palestinian territory':'Palestine',
                       'Holy See':'Vatican',
                       'Republic of Ireland':'Ireland',
                       'The Bahamas':'Bahamas',
                       'The Gambia':'Gambia',
                       'Bahamas, The':'Bahamas',
                       'Gambia, The':'Gambia',
                       'Vatican City':'Vatican',
                       'East Timor':'Timor-Leste',
                       'West Bank and Gaza':'Palestine',
                       'MS Zaandam':'Others',
                       'Diamond Princess':'Others'
                      }
    
    df = df.replace({"country": replace_country})
    # New no. of countries
    print("Number of Countries:", len(df.country.unique()))
    
    '''
    Several 'countries' on the 'df' dataframe have daily data divided into 
    several 'provinces'. Accumulate the 'confirmed', 'deaths' and 'recovered' 
    data for these provinces so that the daily data for each country is only 
    represented by one row by creating a new dataframe 'df_new'.
    '''
    print("Daily Data for Each Country")
    df_new = df.groupby(by=['country', 'date']).sum().reset_index()
    print(df_new)
    
    '''
    Drop all rows in 'df_new' which data 'confirmed' is below 100
    '''
    print("Daily Data for Each Country Which Confimed >= 100")
    df_new = df_new[df_new['confirmed'] >= 100]
    print(df_new)
    
    '''
    Add 'region' and 'climate' columns to 'df_new' and fill in the region and 
    climate for each country by referring to 'df_cat'
    '''
    print("Add 'region' and 'climate' columns")
    df_new = df_new.merge(df_cat, how='left', on="country")
    print(df_new)
    
    '''
    Create a line plot based on dataframe 'df_new' with data 'date' as x and 
    data 'confirmed' as y, where each line represents the accumulative data 
    of each region 
    '''
    # get data for each region
    df_asipac = df_new[df_new['region'] == 'Asia & Pacific'].groupby(by=['date']).sum().reset_index()
    df_europe = df_new[df_new['region'] == 'Europe'].groupby(by=['date']).sum().reset_index()
    df_arab = df_new[df_new['region'] == 'Arab States'].groupby(by=['date']).sum().reset_index()
    df_africa = df_new[df_new['region'] == 'Africa'].groupby(by=['date']).sum().reset_index()
    df_latin = df_new[df_new['region'] == 'South/Latin America'].groupby(by=['date']).sum().reset_index()
    df_northam = df_new[df_new['region'] == 'North America'].groupby(by=['date']).sum().reset_index()
    df_mideast = df_new[df_new['region'] == 'Middle east'].groupby(by=['date']).sum().reset_index()
    
    # plot based on date
    plt.plot(df_asipac["date"], df_asipac["confirmed"], label="Asia & Pacific")
    plt.plot(df_europe["date"], df_europe["confirmed"], label="Europe")
    plt.plot(df_arab["date"], df_arab["confirmed"], label="Arab States")
    plt.plot(df_africa["date"], df_africa["confirmed"], label="Africa")
    plt.plot(df_latin["date"], df_latin["confirmed"], label="South/Latin America")
    plt.plot(df_northam["date"], df_northam["confirmed"], label="North America")
    plt.plot(df_mideast["date"], df_mideast["confirmed"], label="Middle East")
    plt.legend(loc='upper left')
    plt.title("Data Covid-19 Based on Region from Jan to Aug")
    plt.show()
    
    '''
    Create a line plot based on dataframe 'df_new' with data 'date' as x and 
    data 'confirmed' as y, where each line represents the accumulative data 
    of each climate
    '''
    # get data for each climate
    df_nontropic = df_new[df_new['climate'] == 'nontropic'].groupby(by=['date']).sum().reset_index()
    df_tropic = df_new[df_new['climate'] == 'tropic'].groupby(by=['date']).sum().reset_index()
    
    # plot based on date
    plt.plot(df_nontropic["date"], df_nontropic["confirmed"], label="nontropic")
    plt.plot(df_tropic["date"], df_tropic["confirmed"], label="tropic")
    plt.legend(loc='upper left')
    plt.title("Data Covid-19 Based on Climate from Jan to Aug")
    plt.show()
    
    '''
    Create a dataframe 'df_last' that only contains data from the last date of 
    'df_new', where each row shows data for 'confirmed', 'deaths', 'recovered', 
    'region', and 'climate' for each country. 
    '''
    # get last updated data for each country
    print("Last Updated Data for Each Country")
    df_last = df_new.groupby(by=["country"]).max().reset_index()
    print(df_last)
    
    '''
    Based on the df last, identify the top 10 countries with the highest data 
    for 'deaths' and then create the barplot.
    '''
    # identify the top 10 countries with the highest number of deaths
    df_10 = df_last.sort_values(by=['deaths'], ascending=False).head(10)
    print(df_10)
    
    # creating the bar plot 
    plt.bar(df_10["country"], df_10["deaths"], color ='maroon', width = 0.4) 
      
    plt.xlabel("Countries") 
    plt.ylabel("No. of deaths") 
    plt.title("Top 10 Negara dengan Death Tertinggi") 
    plt.show()
    
    '''
    Perform EDA on the 'df_last' dataframe for the 'confirmed', 'deaths' and 
    'recovered' columns using a scatter matrix (distinguish the scatter plot 
    colors by region) 
    '''
    # create a scatter matrix for region
    sns.set_theme(style="ticks")
    ax = sns.pairplot(df_last, hue="region")
    ax.fig.suptitle("Scatter Matrix By Region", y=1.08)
    
    '''
    Perform EDA on the 'df_last' dataframe for the 'confirmed', 'deaths' and 
    'recovered' columns using a scatter matrix (distinguish the scatter plot 
    colors by climate) 
    '''
    # create a scatter matrix for climate
    sns.set_theme(style="ticks")
    ax= sns.pairplot(df_last, hue="climate")
    ax.fig.suptitle("Scatter Matrix Berdasarkan Climate", y=1.08)
    
    '''
    What data do visually appear to have a normal distribution? None
    '''
    # alpha value
    alpha = 0.05
    
    # 'confirmed' normality test
    print("====================confirmed==========================")
    stat, p = stats.normaltest(df_last["confirmed"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Normal Distribution (fail to reject H0)')
    else:
        print('Not Normal Distribution (reject H0)')
        
    # 'deaths' normality test
    print("======================deaths===========================")
    stat, p = stats.normaltest(df_last["deaths"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Normal Distribution (fail to reject H0)')
    else:
        print('Not Normal Distribution (reject H0)')
        
    # 'recovered' normality test
    print("=====================recovered=========================")
    stat, p = stats.normaltest(df_last["recovered"])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Normal Distribution (fail to reject H0)')
    else:
        print('Not Normal Distribution (reject H0)')
        
    '''
    Perform an independent t-test to test whether or not a correlation between 
    'climate' and 'confirmed' in 'df_last'.
    '''
    # convert data type 'climate'
    df_last["climate"] = df_last["climate"].astype('category')
    df_last["climate_cat_codes"] = df_last["climate"].cat.codes
    
    # perform independent t-test
    stat, p = stats.ttest_ind(df_last['confirmed'], df_last['climate_cat_codes'])
    
    '''
    If alpha value = 0.05, can the H0 that 'there is no correlation between the 
    climate group and the' Confirmed 'data be rejected? What is the reason?
    There is a correlation between the climate groups and the data confirmed 
    because the p-value was smaller than the alpha value
    '''
    # hypothesis
    if p > alpha:
        print('fail to reject H0')
    else:
        print('reject H0')
    
    '''
    Build a classifier model to predict region of 'X_new' based on the 
    'confirmed', 'deaths' and 'recovered' data contained in 'df_last'. 
    Show the accuracy of the model using a train-test split.
    '''
    # drop na if exists
    df_last = df_last.dropna()
    
    # train-test split
    X = df_last.iloc[:,2:5]
    y = df_last["region"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify=y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # modeling
    clf = RandomForestClassifier(n_estimators=100, max_features="auto", max_depth=None, 
                                 min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    clf = clf.fit(X_train, y_train)
    
    # predict test data
    y_pred = clf.predict(X_test)
    
    # scoring
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # # parameter grid
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    
    # parameters = {"n_estimators": n_estimators,
    #               "max_features": max_features,
    #               "min_samples_split": max_depth,
    #               "min_samples_leaf": min_samples_split,
    #               "bootstrap": bootstrap
    #              }
    
    # # parameter optimization
    # gs = GridSearchCV(clf, param_grid=parameters, cv=2, verbose=1, n_jobs=-1)
    # gs.fit(X, y)
    
    # # best parameter
    # print("Best Param:", gs.best_params_)
    # print("Best Estimator:", gs.best_estimator_)
    # print("Best Score:", gs.best_score_)
    
    # extract single tree
    estimator = clf.estimators_[0]
    print(estimator)
    
    # visualizing
    fn = ["confirmed", "deaths", "recovered"]
    cn = ['AP', 'EU', 'Arab', 'Afr', 'Latin','NA', 'ME']
    
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=fn,
                               class_names=cn,
                               filled=True, rounded=True,  
                               special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render('dtree_region',view=True)
    # predict new data
    X_new=np.array([[1000,30,200],
                    [2000,40,400],
                    [50,1,2]])
    
    y_pred = clf.predict(X_new)
    print(y_pred)
    
    '''
    Build a classifier model to predict climate of 'X_new' based on the 
    'confirmed', 'deaths' and 'recovered' data contained in 'df_last'. 
    Show the accuracy of the model using a train-test split.
    '''
    # train-test split
    X = df_last.iloc[:,2:5]
    y = df_last["climate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # modeling
    clf = RandomForestClassifier(n_estimators=100, max_features="auto", max_depth=None, 
                                 min_samples_split=2, min_samples_leaf=1, bootstrap=True)
    clf = clf.fit(X_train, y_train)
    
    # predict test data
    y_pred = clf.predict(X_test)
    
    # scoring
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # # parameter grid
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    
    # parameters = {"n_estimators": n_estimators,
    #               "max_features": max_features,
    #               "min_samples_split": max_depth,
    #               "min_samples_leaf": min_samples_split,
    #               "bootstrap": bootstrap
    #              }
    
    # # parameter optimization
    # gs = GridSearchCV(clf, param_grid=parameters, cv=5, verbose=1, n_jobs=-1)
    # gs.fit(X, y)
    
    # # best parameter
    # print("Best Param:", gs.best_params_)
    # print("Best Estimator:", gs.best_estimator_)
    # print("Best Score:", gs.best_score_)
    
    # extract single tree
    estimator = clf.estimators_[0]
    print(estimator)
    
    # visualizing
    fn = ["confirmed", "deaths", "recovered"]
    cn = ['nontropic', 'tropic']
    
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=fn,
                               class_names=cn,
                               filled=True, rounded=True,  
                               special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render('dtree_climate',view=True)
    
    # predict new data
    X_new=np.array([[1000,30,200],
                    [2000,40,400],
                    [50,1,2]])
    
    y_pred = clf.predict(X_new)
    print(y_pred)

    
    '''
    Build a linear regression model for the distribution of the number of 
    deaths in the US from 20 March 2020 - 10 August 2020. Plot this regression 
    model. Calculate the R^2 and RMSE values of the model using a train-test split.
    '''
    # select date ranges
    df_lr = df.set_index('date')
    startdate = pd.to_datetime("2020-03-20").date()
    enddate = pd.to_datetime("2020-08-10").date()
    df_lr = df_lr.loc[startdate : enddate].reset_index()
    
    # train-test split
    X = df_lr[['confirmed']]
    y = df_lr["deaths"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # modeling
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("Intercept:", regressor.intercept_)
    print("Coefficient:", regressor.coef_)
    
    # predict test data
    y_pred = regressor.predict(X_test)
    df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # scoring
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2 Score:', r2_score(y_test, y_pred))
    
    # visualizing
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='red')
    plt.show()
    
    '''
    Build a clustering model with 5 clusters (cluster 0-4) for the 'Z' array. 
    Predict the cluster numbers based on 'confirmed', 'deaths' and 'recovered' 
    data for 'df_last' for the following countries:
        a. Indonesia
        b. Singapore 4
        c. US 1
        d. Italy 4
        e. Iran 4
    '''
    # data
    Z=df_last.loc[:,['confirmed','recovered','deaths']].values
    print(Z[:5])
    
    # normalization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(Z)
    
    # modeling
    kmeans = KMeans(init="random", n_clusters=5, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_features)
    print(kmeans.labels_[:5])
    
    # scoring
    kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(2)
    print("Silhouette Score:", kmeans_silhouette)
    
    # predict new data
    df_indonesia = df_last.loc[df_last['country'] == "Indonesia"].iloc[:,2:5]
    print(df_indonesia)
    
    df_singapore = df_last.loc[df_last['country'] == "Singapore"].iloc[:,2:5]
    print(df_singapore)
    
    df_us = df_last.loc[df_last['country'] == "US"].iloc[:,2:5]
    print(df_us)
    
    df_italy = df_last.loc[df_last['country'] == "Italy"].iloc[:,2:5]
    print(df_italy)
    
    df_iran = df_last.loc[df_last['country'] == "Iran"].iloc[:,2:5]
    print(df_iran)
    
    df_new_pred_cluster = pd.concat([df_indonesia, df_singapore, df_us, df_italy, df_iran])
    
    X = StandardScaler().fit_transform(df_new_pred_cluster)
    y_pred = kmeans.predict(X)
    print(y_pred)