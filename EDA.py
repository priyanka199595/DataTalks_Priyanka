import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
#import plotly.figure_factory as ff
import time


from datetime import datetime
import seaborn as sns
sns.set_style("whitegrid")
import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numexpr as ne

import xgboost as xgb
from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV


def get_EDA_page():
    if not os.path.isfile("Data/NetflixRatings.csv"):
        startTime = datetime.now()
        data = open("Data/NetflixRatings.csv", mode = "w") 
        files = ['Data/combined_data_4.txt']
        for file in files:
            print("Reading from file: "+str(file)+"...")
            with open(file) as f:  
                for line in f:
                    line = line.strip() 
                    if line.endswith(":"):
                        movieID = line.replace(":", "")
                    else:
                        row = [] 
                        row = [x for x in line.split(",")] #custID, rating and date are separated by comma
                        row.insert(0, movieID)
                        data.write(",".join(row))
                        data.write("\n")
            print("Reading of file: "+str(file)+" is completed\n")
        data.close()
        print("Total time taken for execution of this code = "+str(datetime.now() - startTime))

    else:
        print("data is already loaded")

    # creating data frame from our output csv file.
    if not os.path.isfile("Data/NetflixData.pkl"):
        startTime = datetime.now()
        Final_Data = pd.read_csv("Data/NetflixRatings.csv", sep=",", names = ["MovieID","CustID", "Ratings", "Date"])
        Final_Data["Date"] = pd.to_datetime(Final_Data["Date"])
        Final_Data.sort_values(by = "Date", inplace = True)
        print("Time taken for execution of above code = "+str(datetime.now() - startTime))
        st.write("data frame created")
    else:
        print("data frame already present")

    # storing pandas dataframe as a picklefile for later use
    if not os.path.isfile("Data/NetflixData.pkl"):
        Final_Data.to_pickle("Data/NetflixData.pkl")
        st.write("pkl created")
    else:
        Final_Data = pd.read_pickle("Data/NetflixData.pkl")
        print("pkl already present")

    if st.checkbox("Show Final_Data"):
        st.write(Final_Data)
        if st.checkbox("Show all the column Names"):
            st.write(Final_Data.columns)

########
    if st.checkbox("Show size of dataset"):
        if st.checkbox("Show row size"):
            st.write(Final_Data.shape[0])
        if st.checkbox("Show column size"):
            st.write(Final_Data.shape[1])
        if st.checkbox("Show complete dataset size"):
            st.write(Final_Data.shape)
        if st.checkbox("Show desc of Ratings in final data"):
            Final_Data.describe()["Ratings"]    

    st.write("**displaying final dataset header lines using area chart**")
    st.area_chart(Final_Data)


    print("Number of NaN values = "+str(Final_Data.isnull().sum()))

    duplicates = Final_Data.duplicated(["MovieID","CustID", "Ratings"])
    print("Number of duplicate rows = "+str(duplicates.sum()))

#####
    if st.checkbox("Show unique customer & movieId in Total Data:"):
        st.write("Total number of movie ratings = ", str(Final_Data.shape[0]))
        st.write("Number of unique users = ", str(len(np.unique(Final_Data["CustID"]))))
        st.write("Number of unique movies = ", str(len(np.unique(Final_Data["MovieID"]))))
######### creating pkl file
    if not os.path.isfile("Data/TrainData.pkl"):
        Final_Data.iloc[:int(Final_Data.shape[0]*0.80)].to_pickle("Data/TrainData.pkl")
        Train_Data = pd.read_pickle("Data/TrainData.pkl")
        Train_Data.reset_index(drop = True, inplace = True)
    else:
        Train_Data = pd.read_pickle("Data/TrainData.pkl")
        Train_Data.reset_index(drop = True, inplace = True)

    if not os.path.isfile("Data/TestData.pkl"):
        Final_Data.iloc[int(Final_Data.shape[0]*0.80):].to_pickle("Data/TestData.pkl")
        Test_Data = pd.read_pickle("Data/TestData.pkl")
        Test_Data.reset_index(drop = True, inplace = True)
    else:
        Test_Data = pd.read_pickle("Data/TestData.pkl")
        Test_Data.reset_index(drop = True, inplace = True)
#########

    if st.checkbox("Showing dataset of Train_Data & Test_Data"):
        st.area_chart(Train_Data)
        st.area_chart(Test_Data)

    if st.checkbox("Show unique customer & movieId in Train DataSet:"):
        st.write("Total number of movie ratings in train data = ", str(Train_Data.shape[0]))
        st.write("Number of unique users in train data = ", str(len(np.unique(Train_Data["CustID"]))))
        st.write("Number of unique movies in train data = ", str(len(np.unique(Train_Data["MovieID"]))))
        st.write("Highest value of a User ID = ", str(max(Train_Data["CustID"].values)))
        st.write("Highest value of a Movie ID =  ", str(max(Train_Data["MovieID"].values)))


    if st.checkbox("Show unique customer & movieId in Test DataSet:"):
        st.write("Total number of movie ratings in Test data = ", str(Test_Data.shape[0]))
        st.write("Number of unique users in Test data = ", str(len(np.unique(Test_Data["CustID"]))))
        st.write("Number of unique movies in trTestain data = ", str(len(np.unique(Test_Data["MovieID"]))))
        st.write("Highest value of a User ID = ", str(max(Test_Data["CustID"].values)))
        st.write("Highest value of a Movie ID =  ", str(max(Test_Data["MovieID"].values)))

    ##########
    
    def changingLabels(number):
        return str(number/10**6) + "M"

    plt.figure(figsize = (12, 8))
    ax = sns.countplot(x="Ratings", data=Train_Data)

    ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])

    plt.tick_params(labelsize = 15)
    plt.title("Distribution of Ratings in train data", fontsize = 20)
    plt.xlabel("Ratings", fontsize = 20)
    plt.ylabel("Number of Ratings(Millions)", fontsize = 20)
    st.pyplot()
    st.write("This graph will  show how **Distribution of Ratings** which shows the overall maturity level of the whole series and is provided by the audience :smile: ")

    Train_Data["DayOfWeek"] = Train_Data.Date.dt.weekday_name
    plt.figure(figsize = (10,8))
    ax = Train_Data.resample("M", on = "Date")["Ratings"].count().plot()
    ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
    ax.set_title("Number of Ratings per Month", fontsize = 20)
    ax.set_xlabel("Date", fontsize = 20)
    ax.set_ylabel("Number of Ratings Per Month(Millions)", fontsize = 20)
    plt.tick_params(labelsize = 15)
    st.pyplot()
    st.write("This Graph will represents the **Number of Ratings Per Month** means counts of ratings grouped by months :smile:")

    st.write("**Analysis of Ratings given by user**")
    no_of_rated_movies_per_user = Train_Data.groupby(by = "CustID")["Ratings"].count().sort_values(ascending = False)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14,7))
    sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, ax = axes[0])
    axes[0].set_title("Fig1", fontsize = 18)
    axes[0].set_xlabel("Number of Ratings by user", fontsize = 18)
    axes[0].tick_params(labelsize = 15)
    sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, cumulative = True, ax = axes[1])
    axes[1].set_title("Fig2", fontsize = 18)
    axes[1].set_xlabel("Number of Ratings by user", fontsize = 18)
    axes[1].tick_params(labelsize = 15)
    fig.subplots_adjust(wspace=2)
    plt.tight_layout()
    st.pyplot()

    ####
    st.write("Above fig1 graph shows that almost all of the users give very few ratings. There are very **few users who's ratings count is high** .Similarly, above fig2 graph shows that **almost 99% of users give very few ratings**")
    quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01))
    fig = plt.figure(figsize = (10, 6))
    axes = fig.add_axes([0.1,0.1,1,1])
    axes.set_title("Quantile values of Ratings Per User", fontsize = 20)
    axes.set_xlabel("Quantiles", fontsize = 20)
    axes.set_ylabel("Ratings Per User", fontsize = 20)
    axes.plot(quantiles)
    plt.scatter(x = quantiles.index[::5], y = quantiles.values[::5], c = "blue", s = 70, label="quantiles with 0.05 intervals")
    plt.scatter(x = quantiles.index[::25], y = quantiles.values[::25], c = "red", s = 70, label="quantiles with 0.25 intervals")
    plt.legend(loc='upper left', fontsize = 20)
    for x, y in zip(quantiles.index[::25], quantiles.values[::25]):
        plt.annotate(s = '({},{})'.format(x, y), xy = (x, y), fontweight='bold', fontsize = 16, xytext=(x-0.05, y+180))
    axes.tick_params(labelsize = 15)
    st.pyplot()

    st.write("this graph shows the Quantile values of Ratings Per User")
    st.write("**Analysis of Ratings Per Movie** :smile:")
    no_of_ratings_per_movie = Train_Data.groupby(by = "MovieID")["Ratings"].count().sort_values(ascending = False)
    fig = plt.figure(figsize = (12, 6))
    axes = fig.add_axes([0.1,0.1,1,1])
    plt.title("Number of Ratings Per Movie", fontsize = 20)
    plt.xlabel("Movie", fontsize = 20)
    plt.ylabel("Count of Ratings", fontsize = 20)
    plt.plot(no_of_ratings_per_movie.values)
    plt.tick_params(labelsize = 15)
    axes.set_xticklabels([])
    st.pyplot()

    st.write("This graph shows the number of rating(in count) each movie achieved by the audience, which clearly shows that there are some movies which are very popular and were rated by many users as comapared to other movies ")
    st.write("**Analysis of Movie Ratings on Day of Week** :smile:")
    fig = plt.figure(figsize = (12, 8))
    axes = sns.countplot(x = "DayOfWeek", data = Train_Data)
    axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
    axes.set_xlabel("Day of Week", fontsize = 20)
    axes.set_ylabel("Number of Ratings", fontsize = 20)
    axes.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
    axes.tick_params(labelsize = 15)
    st.pyplot()

    st.write("This graph will show Analysis of Movie Ratings on Day of Week in bar graph format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")
    fig = plt.figure(figsize = (12, 8))
    axes = sns.boxplot(x = "DayOfWeek", y = "Ratings", data = Train_Data)
    axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
    axes.set_xlabel("Day of Week", fontsize = 20)
    axes.set_ylabel("Number of Ratings", fontsize = 20)
    axes.tick_params(labelsize = 15)
    st.pyplot()

    st.write("This graph will show Analysis of Movie Ratings on Day of Week in box plot format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")
    average_ratings_dayofweek = Train_Data.groupby(by = "DayOfWeek")["Ratings"].mean()
    st.write("**Average Ratings on Day of Weeks**")
    st.write(average_ratings_dayofweek)
    st.write("**This Average Ratings on Day of Weeks will represented in graphical format** ")
    st.area_chart(average_ratings_dayofweek)
    st.write("this graph represents that average rating is mostly lies between 3 to 4.")
    st.write("**Distribution of Movie ratings amoung Users**")
    plt.scatter(Test_Data["CustID"],Test_Data["MovieID"])
    st.pyplot()


####################Creating USER-ITEM sparse matrix from data frame

    startTime = datetime.now()
    print("Creating USER_ITEM sparse matrix for train Data")
    if os.path.isfile("Data/TrainUISparseData.npz"):
        print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
        TrainUISparseData = sparse.load_npz("Data/TrainUISparseData.npz")
        print("Shape of Train Sparse matrix = "+str(TrainUISparseData.shape))
    
    else:
        print("We are creating sparse data")
        TrainUISparseData = sparse.csr_matrix((Train_Data.Ratings, (Train_Data.CustID, Train_Data.MovieID)))
        print("Creation done. Shape of sparse matrix = "+str(TrainUISparseData.shape))
        print("Saving it into disk for furthur usage.")
        sparse.save_npz("Data/TrainUISparseData.npz", TrainUISparseData)
        print("Done\n")

    print(datetime.now() - startTime)

###############Creating USER-ITEM sparse matrix from data frame for test data

    startTime = datetime.now()
    print("Creating USER_ITEM sparse matrix for test Data")
    if os.path.isfile("Data/TestUISparseData.npz"):
        print("Sparse Data is already present in your disk, no need to create further. Loading Sparse Matrix")
        TestUISparseData = sparse.load_npz("Data/TestUISparseData.npz")
        print("Shape of Test Sparse Matrix = "+str(TestUISparseData.shape))
    else:
        print("We are creating sparse data")
        TestUISparseData = sparse.csr_matrix((Test_Data.Ratings, (Test_Data.CustID, Test_Data.MovieID)))
        print("Creation done. Shape of sparse matrix = "+str(TestUISparseData.shape))
        print("Saving it into disk for furthur usage.")
        sparse.save_npz("Data/TestUISparseData.npz", TestUISparseData)
        print("Done\n")

    print(datetime.now() - startTime)



    rows,cols = TrainUISparseData.shape
    presentElements = TrainUISparseData.count_nonzero()

    print("Sparsity Of Train matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))

    rows,cols = TestUISparseData.shape
    presentElements = TestUISparseData.count_nonzero()

    print("Sparsity Of Test matrix : {}% ".format((1-(presentElements/(rows*cols)))*100))

    #################Finding Global average of all movie ratings, Average rating per user, and Average rating per movie


    def getAverageRatings(sparseMatrix, if_user):
        ax = 1 if if_user else 0
        #axis = 1 means rows and axis = 0 means columns 
        sumOfRatings = sparseMatrix.sum(axis = ax).A1  #this will give an array of sum of all the ratings of user if axis = 1 else 
        #sum of all the ratings of movies if axis = 0
        noOfRatings = (sparseMatrix!=0).sum(axis = ax).A1  #this will give a boolean True or False array, and True means 1 and False 
        #means 0, and further we are summing it to get the count of all the non-zero cells means length of non-zero cells
        rows, cols = sparseMatrix.shape
        averageRatings = {i: sumOfRatings[i]/noOfRatings[i] for i in range(rows if if_user else cols) if noOfRatings[i]!=0}
        return averageRatings

    Global_Average_Rating = TrainUISparseData.sum()/TrainUISparseData.count_nonzero()
    print("Global Average Rating {}".format(Global_Average_Rating))

    AvgRatingUser = getAverageRatings(TrainUISparseData, True)


    #############Machine Learning Models

    def get_sample_sparse_matrix(sparseMatrix, n_users, n_movies):
        startTime = datetime.now()
        users, movies, ratings = sparse.find(sparseMatrix)
        uniq_users = np.unique(users)
        uniq_movies = np.unique(movies)
        np.random.seed(15)   #this will give same random number everytime, without replacement
        userS = np.random.choice(uniq_users, n_users, replace = True)
        movieS = np.random.choice(uniq_movies, n_movies, replace = True)
        mask = np.logical_and(np.isin(users, userS), np.isin(movies, movieS))
        sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])), 
                                                     shape = (max(userS)+1, max(movieS)+1))
        print("Sparse Matrix creation done. Saving it for later use.")
        sparse.save_npz(path, sparse_sample)
        print("Done")
        print("Shape of Sparse Sampled Matrix = "+str(sparse_sample.shape))
    
        print(datetime.now() -startTime)
        return sparse_sample
    
    ####Creating Sample Sparse Matrix for Train Data

    path = "Data/TrainUISparseData_Sample.npz"
    if not os.path.isfile(path):
        print("Sample sparse matrix is not present in the disk. We are creating it...")
        train_sample_sparse = get_sample_sparse_matrix(TrainUISparseData, 4000, 400)
    else:
        print("File is already present in the disk. Loading the file...")
        train_sample_sparse = sparse.load_npz(path)
        print("File loading done.")
        print("Shape of Train Sample Sparse Matrix = "+str(train_sample_sparse.shape))

    ##########Creating Sample Sparse Matrix for Test Data

    path = "Data/TestUISparseData_Sample.npz"
    if not os.path.isfile(path):
        print("Sample sparse matrix is not present in the disk. We are creating it...")
        test_sample_sparse = get_sample_sparse_matrix(TestUISparseData, 2000, 200)
    else:
        print("File is already present in the disk. Loading the file...")
        test_sample_sparse = sparse.load_npz(path)
        print("File loading done.")
        print("Shape of Test Sample Sparse Matrix = "+str(test_sample_sparse.shape))
    #####print("Global average of all movies ratings in Train Sample Sparse is {}".format(np.round((train_sample_sparse.sum()/train_sample_sparse.count_nonzero()), 2)))
    globalAvgMovies = getAverageRatings(train_sample_sparse, False)
    globalAvgUsers = getAverageRatings(train_sample_sparse, True)

    #######   Featurizing data for regression problem
    ###### Featurizing Train Data

    sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(train_sample_sparse)


    if os.path.isfile("Data/Train_Regression.csv"):
        print("File is already present in your disk. You do not have to prepare it again.")
    else:
        startTime = datetime.now()
        print("Preparing Train csv file for {} rows".format(len(sample_train_ratings)))
        with open("Data/Train_Regression.csv", mode = "w") as data:
            count = 0
            for user, movie, rating in zip(sample_train_users, sample_train_movies, sample_train_ratings):
                row = list()
                row.append(user)  #appending user ID
                row.append(movie) #appending movie ID
                row.append(train_sample_sparse.sum()/train_sample_sparse.count_nonzero()) #appending global average rating

#----------------------------------Ratings given to "movie" by top 5 similar users with "user"--------------------#
                similar_users = cosine_similarity(train_sample_sparse[user], train_sample_sparse).ravel()
                similar_users_indices = np.argsort(-similar_users)[1:]
                similar_users_ratings = train_sample_sparse[similar_users_indices, movie].toarray().ravel()
                top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
                top_similar_user_ratings.extend([globalAvgMovies[movie]]*(5-len(top_similar_user_ratings)))
                #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "movie" average
                #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "movie" average rating.
                row.extend(top_similar_user_ratings)
            
 #----------------------------------Ratings given by "user" to top 5 similar movies with "movie"------------------#similar_movies = cosine_similarity(train_sample_sparse[:,movie].T, train_sample_sparse.T).ravel()
                similar_movies_indices = np.argsort( -similar_movies)[1:]
                similar_movies_ratings = train_sample_sparse[user, similar_movies_indices].toarray().ravel()
                top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
                top_similar_movie_ratings.extend([globalAvgUsers[user]]*(5-len(top_similar_movie_ratings)))
                #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "user" average
                #rating. Let say only 3 out of 5 ratings are available then rest 2 will be "user" average rating.
                row.extend(top_similar_movie_ratings)
            
 #----------------------------------Appending "user" average, "movie" average & rating of "user""movie"-----------#
                row.append(globalAvgUsers[user])
                row.append(globalAvgMovies[movie])
                row.append(rating)
            
#-----------------------------------Converting rows and appending them as comma separated values to csv file------#
                data.write(",".join(map(str, row)))
                data.write("\n")
                count += 1
                if count % 2000 == 0:
                    print("Done for {}. Time elapsed: {}".format(count, (datetime.now() - startTime)))
                
        print("Total Time for {} rows = {}".format(len(sample_train_ratings), (datetime.now() - startTime)))
################
    Train_Reg = pd.read_csv("Data/Train_Regression.csv", names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])
    #Train_Reg.head()   
    ########    Featurizing Test Data    #####################3

    sample_test_users, sample_test_movies, sample_test_ratings = sparse.find(test_sample_sparse)
    if os.path.isfile("Data/Test_Regression.csv"):
        print("File is already present in your disk. You do not have to prepare it again.")
    else:
        startTime = datetime.now()
        print("Preparing Test csv file for {} rows".format(len(sample_test_ratings)))
        with open("Data/Test_Regression.csv", mode = "w") as data:
            count = 0
            for user, movie, rating in zip(sample_test_users, sample_test_movies, sample_test_ratings):
                row = list()
                row.append(user)  #appending user ID
                row.append(movie) #appending movie ID
                row.append(train_sample_sparse.sum()/train_sample_sparse.count_nonzero()) #appending global average rating#-----------------------------Ratings given to "movie" by top 5 similar users with "user"-------------------------#
                try:
                    similar_users = cosine_similarity(train_sample_sparse[user], train_sample_sparse).ravel()
                    similar_users_indices = np.argsort(-similar_users)[1:]
                    similar_users_ratings = train_sample_sparse[similar_users_indices, movie].toarray().ravel()
                    top_similar_user_ratings = list(similar_users_ratings[similar_users_ratings != 0][:5])
                    top_similar_user_ratings.extend([globalAvgMovies[movie]]*(5-len(top_similar_user_ratings)))
                    #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "movie" 
                    #average rating. Let say only 3 out of 5 ratings are available then rest 2 will be "movie" average rating.
                    row.extend(top_similar_user_ratings)
                #########Cold Start Problem, for a new user or a new movie#########    
                except(IndexError, KeyError):
                    global_average_train_rating = [train_sample_sparse.sum()/train_sample_sparse.count_nonzero()]*5
                    row.extend(global_average_train_rating)
                except:
                    raise
                
 #-----------------------------Ratings given by "user" to top 5 similar movies with "movie"-----------------------#
                try:
                    similar_movies = cosine_similarity(train_sample_sparse[:,movie].T, train_sample_sparse.T).ravel()
                    similar_movies_indices = np.argsort(-similar_movies)[1:]
                    similar_movies_ratings = train_sample_sparse[user, similar_movies_indices].toarray().ravel()
                    top_similar_movie_ratings = list(similar_movies_ratings[similar_movies_ratings != 0][:5])
                    top_similar_movie_ratings.extend([globalAvgUsers[user]]*(5-len(top_similar_movie_ratings)))
                    #above line means that if top 5 ratings are not available then rest of the ratings will be filled by "user" 
                    #average rating. Let say only 3 out of 5 ratings are available then rest 2 will be "user" average rating.
                    row.extend(top_similar_movie_ratings)
                #########Cold Start Problem, for a new user or a new movie#########
                except(IndexError, KeyError):
                    global_average_train_rating = [train_sample_sparse.sum()/train_sample_sparse.count_nonzero()]*5
                    row.extend(global_average_train_rating)
                except:
                    raise
                
 #-----------------------------Appending "user" average, "movie" average & rating of "user""movie"----------------#try:        
                try: 
                    row.append(globalAvgUsers[user])
                except (KeyError):
                    global_average_train_rating = train_sample_sparse.sum()/train_sample_sparse.count_nonzero()
                    row.append(global_average_train_rating)
                except:
                    raise
                
                try:
                    row.append(globalAvgMovies[movie])
                except(KeyError):
                    global_average_train_rating = train_sample_sparse.sum()/train_sample_sparse.count_nonzero()
                    row.append(global_average_train_rating)
                except:
                    raise
                
                row.append(rating)
            
#------------------------------Converting rows and appending them as comma separated values to csv file-----------#
                data.write(",".join(map(str, row)))
                data.write("\n")

                count += 1
                if count % 100 == 0:
                    print("Done for {}. Time elapsed: {}".format(count, (datetime.now() - startTime)))
                
        print("Total Time for {} rows = {}".format(len(sample_test_ratings), (datetime.now() - startTime)))

    Test_Reg = pd.read_csv("Data/Test_Regression.csv", names = ["User_ID", "Movie_ID", "Global_Average", "SUR1", "SUR2", "SUR3", "SUR4", "SUR5", "SMR1", "SMR2", "SMR3", "SMR4", "SMR5", "User_Average", "Movie_Average", "Rating"])
    #Test_Reg.head()

    ##
    ###### Transforming Data for Surprise Models 
    Train_Reg[['User_ID', 'Movie_ID', 'Rating']].head(5)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(Train_Reg[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset = data.build_full_trainset() 
 
    testset = list(zip(Test_Reg["User_ID"].values, Test_Reg["Movie_ID"].values, Test_Reg["Rating"].values))


    error_table = pd.DataFrame(columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"])
    model_train_evaluation = dict()
    model_test_evaluation = dict()

    def make_table(model_name, rmse_train, mape_train, rmse_test, mape_test):
        global error_table
        #All variable assignments in a function store the value in the local symbol table; whereas variable references first look 
        #in the local symbol table, then in the global symbol table, and then in the table of built-in names. Thus, global variables 
        #cannot be directly assigned a value within a function (unless named in a global statement), 
        #although they may be referenced.
        error_table = error_table.append(pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"]))
        error_table.reset_index(drop = True, inplace = True)


    ###### Utility Functions for Regression Models
    def error_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(abs((y_true - y_pred)/y_true))*100
        return rmse, mape

    def train_test_xgboost(x_train, x_test, y_train, y_test, model_name):
        startTime = datetime.now()
        train_result = dict()
        test_result = dict()
    
        clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = 10)
        clf.fit(x_train, y_train)
    
        print("-"*50)
        print("TRAIN DATA")
        y_pred_train = clf.predict(x_train)
        rmse_train, mape_train = error_metrics(y_train, y_pred_train)
        print("RMSE = {}".format(rmse_train))
        print("MAPE = {}".format(mape_train))
        print("-"*50)
        train_result = {"RMSE": rmse_train, "MAPE": mape_train, "Prediction": y_pred_train}
    
        print("TEST DATA")
        y_pred_test = clf.predict(x_test)
        rmse_test, mape_test = error_metrics(y_test, y_pred_test)
        print("RMSE = {}".format(rmse_test))
        print("MAPE = {}".format(mape_test))
        print("-"*50)
        test_result = {"RMSE": rmse_test, "MAPE": mape_test, "Prediction": y_pred_test}
        
        print("Time Taken = "+str(datetime.now() - startTime))
    
        plot_importance(xgb, clf)
    
        make_table(model_name, rmse_train, mape_train, rmse_test, mape_test)
    
        return train_result, test_result
    #######################
    def plot_importance(model, clf):
        fig = plt.figure(figsize = (4, 3))
        ax = fig.add_axes([0,0,1,1])
        model.plot_importance(clf, ax = ax, height = 0.3)
        ax.set_xlabel("F Score", fontsize = 20)
        ax.set_ylabel("Features", fontsize = 20)
        ax.set_title("Feature Importance", fontsize = 20)
        #ax.set_tick_params(labelsize = 15)
        st.pyplot(fig = fig)
        #plt.show()

    #st.plotly_chart(fig,use_container_width=True)
 
    ###### Utility Functions for Surprise Models

    def get_ratings(predictions):
        actual = np.array([pred.r_ui for pred in predictions])
        predicted = np.array([pred.est for pred in predictions])
        return actual, predicted
    #in surprise prediction of every data point is returned as dictionary like this:
    #"user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}"
    #In this dictionary, "r_ui" is a key for actual rating and "est" is a key for predicted rating
    def get_error(predictions):
        actual, predicted = get_ratings(predictions)
        rmse = np.sqrt(mean_squared_error(actual, predicted)) 
        mape = np.mean(abs((actual - predicted)/actual))*100
        return rmse, mape


    my_seed = 15
    random.seed(my_seed)
    np.random.seed(my_seed)

    def run_surprise(algo, trainset, testset, model_name):
        startTime = datetime.now()
    
        train = dict()
        test = dict()
    
        algo.fit(trainset)
        #You can check out above function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
        #"Train-test split and the fit() method" section
    
#-----------------Evaluating Train Data------------------#
        print("-"*50)
        print("TRAIN DATA")
        train_pred = algo.test(trainset.build_testset())
        #You can check out "algo.test()" function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
        #"Train-test split and the fit() method" section
        #You can check out "trainset.build_testset()" function at "https://surprise.readthedocs.io/en/stable/FAQ.html#can-i-use-my-own-dataset-with-surprise-and-can-it-be-a-pandas-dataframe" in 
        #"How to get accuracy measures on the training set" section
        train_actual, train_predicted = get_ratings(train_pred)
        train_rmse, train_mape = get_error(train_pred)
        print("RMSE = {}".format(train_rmse))
        print("MAPE = {}".format(train_mape))
        print("-"*50)
        train = {"RMSE": train_rmse, "MAPE": train_mape, "Prediction": train_predicted}
    
#-----------------Evaluating Test Data------------------#
        print("TEST DATA")
        test_pred = algo.test(testset)
        #You can check out "algo.test()" function at "https://surprise.readthedocs.io/en/stable/getting_started.html" in 
        #"Train-test split and the fit() method" section
        test_actual, test_predicted = get_ratings(test_pred)
        test_rmse, test_mape = get_error(test_pred)
        print("RMSE = {}".format(test_rmse))
        print("MAPE = {}".format(test_mape))
        print("-"*50)
        test = {"RMSE": test_rmse, "MAPE": test_mape, "Prediction": test_predicted}
    
        print("Time Taken = "+str(datetime.now() - startTime))
    
        make_table(model_name, train_rmse, train_mape, test_rmse, test_mape)
    
        return train, test    
    ##
    ################## XGBoost 13 Features################### 
    x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    y_train = Train_Reg["Rating"]

    y_test = Test_Reg["Rating"]

    train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGBoost_13")

    model_train_evaluation["XGBoost_13"] = train_result
    model_test_evaluation["XGBoost_13"] = test_result

####################################################
###################   2. Surprise BaselineOnly Model    #################################
    bsl_options = {"method":"sgd", "learning_rate":0.01, "n_epochs":25}

    algo = BaselineOnly(bsl_options=bsl_options)
    #You can check the docs of above used functions at:https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#baseline-estimates-configuration
    #at section "Baselines estimates configuration".

    train_result, test_result = run_surprise(algo, trainset, testset, "BaselineOnly")

    model_train_evaluation["BaselineOnly"] = train_result
    model_test_evaluation["BaselineOnly"] = test_result

############# 3. XGBoost 13 Features + Surprise BaselineOnly Model  ####################
    Train_Reg["BaselineOnly"] = model_train_evaluation["BaselineOnly"]["Prediction"]
    Test_Reg["BaselineOnly"] = model_test_evaluation["BaselineOnly"]["Prediction"]

    x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    y_train = Train_Reg["Rating"]

    y_test = Test_Reg["Rating"]

    train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL")

    model_train_evaluation["XGB_BSL"] = train_result
    model_test_evaluation["XGB_BSL"] = test_result

    ################### 4. Surprise KNN-Baseline with User-User and Item-Item Similarity     #########
    param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [True], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}

    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # best RMSE score
    #print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])

    #######   Applying KNNBaseline User-User with best parameters    ########
    sim_options = {'name':'pearson_baseline', 'user_based':True, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

    bsl_options = {'method': 'sgd'} 

    algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)

    train_result, test_result = run_surprise(algo, trainset, testset, "KNNBaseline_User")

    model_train_evaluation["KNNBaseline_User"] = train_result
    model_test_evaluation["KNNBaseline_User"] = test_result

    ##########  4.2 Surprise KNN-Baseline with Item-Item    #############

    param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [False], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}

    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # best RMSE score
    #print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])

    ###############  Applying KNNBaseline Item-Item with best parameters  ######
    sim_options = {'name':'pearson_baseline', 'user_based':False, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

    bsl_options = {'method': 'sgd'} 

    algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)

    train_result, test_result = run_surprise(algo, trainset, testset, "KNNBaseline_Item")

    model_train_evaluation["KNNBaseline_Item"] = train_result
    model_test_evaluation["KNNBaseline_Item"] = test_result
    ###########   5. XGBoost 13 Features + Surprise BaselineOnly + Surprise KNN Baseline    ###############
    Train_Reg["KNNBaseline_User"] = model_train_evaluation["KNNBaseline_User"]["Prediction"]
    Train_Reg["KNNBaseline_Item"] = model_train_evaluation["KNNBaseline_Item"]["Prediction"]

    Test_Reg["KNNBaseline_User"] = model_test_evaluation["KNNBaseline_User"]["Prediction"]
    Test_Reg["KNNBaseline_Item"] = model_test_evaluation["KNNBaseline_Item"]["Prediction"]

    #st.write(Train_Reg.head())

    x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    y_train = Train_Reg["Rating"]

    y_test = Test_Reg["Rating"]

    train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN")

    model_train_evaluation["XGB_BSL_KNN"] = train_result
    model_test_evaluation["XGB_BSL_KNN"] = test_result
    ##
    #########################################################################################################
    #################   6. Matrix Factorization SVD    ################################

    param_grid  = {'n_factors': [5,7,10,15,20,25,35,50,70,90]}   #here, n_factors is the equivalent to dimension 'd' when matrix 'A'
    #is broken into 'b' and 'c'. So, matrix 'A' will be of dimension n*m. So, matrices 'b' and 'c' will be of dimension n*d and m*d.

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # best RMSE score
    #print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])

    #############   Applying SVD with best parameters   #################

    algo = SVD(n_factors = gs.best_params['rmse']['n_factors'], biased=True, verbose=True)

    train_result, test_result = run_surprise(algo, trainset, testset, "SVD")

    model_train_evaluation["SVD"] = train_result
    model_test_evaluation["SVD"] = test_result

    #############   7. Matrix Factorization SVDpp with implicit feedback    ############

    param_grid = {'n_factors': [10, 30, 50, 80, 100], 'lr_all': [0.002, 0.006, 0.018, 0.054, 0.10]}

    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # best RMSE score
    #print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    #print(gs.best_params['rmse'])

    ##########
    algo = SVDpp(n_factors = gs.best_params['rmse']['n_factors'], lr_all = gs.best_params['rmse']["lr_all"], verbose=True)

    train_result, test_result = run_surprise(algo, trainset, testset, "SVDpp")

    model_train_evaluation["SVDpp"] = train_result
    model_test_evaluation["SVDpp"] = test_result

    ############## 8. XGBoost 13 Features + Surprise BaselineOnly + Surprise KNN Baseline + SVD + SVDpp

    Train_Reg["SVD"] = model_train_evaluation["SVD"]["Prediction"]
    Train_Reg["SVDpp"] = model_train_evaluation["SVDpp"]["Prediction"]

    Test_Reg["SVD"] = model_test_evaluation["SVD"]["Prediction"]
    Test_Reg["SVDpp"] = model_test_evaluation["SVDpp"]["Prediction"]

    #######
    x_train = Train_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    x_test = Test_Reg.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

    y_train = Train_Reg["Rating"]

    y_test = Test_Reg["Rating"]

    train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN_MF")

    model_train_evaluation["XGB_BSL_KNN_MF"] = train_result
    model_test_evaluation["XGB_BSL_KNN_MF"] = test_result

    ########## 9. Surprise KNN Baseline + SVD + SVDpp  ###################

    x_train = Train_Reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]

    x_test = Test_Reg[["KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp"]]

    y_train = Train_Reg["Rating"]

    y_test = Test_Reg["Rating"]

    train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_KNN_MF")

    model_train_evaluation["XGB_KNN_MF"] = train_result
    model_test_evaluation["XGB_KNN_MF"] = test_result

    ###########################

    error_table2 = error_table.drop(["Train MAPE", "Test MAPE"], axis = 1)

    error_table2.plot(x = "Model", kind = "bar", figsize = (14, 8), grid = True, fontsize = 15)
    plt.title("Train and Test RMSE and MAPE of all Models", fontsize = 20)
    plt.ylabel("Error Values", fontsize = 20)
    plt.legend(bbox_to_anchor=(1, 1), fontsize = 20)
    st.pyplot()
    #plt.show()

    #########
    error_table.drop(["Train MAPE", "Test MAPE"], axis = 1).style.highlight_min(axis=0)











