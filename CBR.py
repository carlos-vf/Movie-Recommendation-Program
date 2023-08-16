# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:09:30 2022

@author: 

The program reads and stores two documents (.json file with all the information
about all films, .csv file with the titles of the films and their rating (1 or 5)).

In order to check if the recommendation system works:
    - The rated movie set is divided in n parts
    - For each movie in each part:
        - Calculate the similarity between this movie and all the others
        - Take the k most similar movies
        - If half or more are liked:
            - Recommend the movie
        - Else:
            - Don't recommend the movie
        - Check if the result matches with user's rating
    - Calculate the success rate
    
Similarity function:
    The function returns a global score. The global score is calculated
    with the partial scores and the weighs. Each feature has a score between
    0 and 1, and a weight between 1 and 6 depending of its importance.
    Lower score = higher similarity
    
"""
import json
import numpy as np
import copy
import math
import random

NUM_OF_PARTS = 10
NUM_OF_NEIGHBORS = 5
SHUFFLE = True
# Default weights
weights = {'runtime':2,
           'language':3,
           'release_year':6,
           'genre':1,
           'actors':5,
           'directors':6,
           'keywords':3
           }
        
def main(jsonFile, csvFile, weights=weights, printAll=True):
    
    # Dictionary with the information of all movies
    dataset = readInformation(jsonFile)
    
    # Dictionary with the rating if each movie
    ratings = readRatings(csvFile)
    
    # Titles from all rated movies
    titles = list(ratings.keys())
    
    # Divide the set in n parts
    # n = 5
    if SHUFFLE:
        random.shuffle(titles)
    chunkedArrays = np.array_split(titles, NUM_OF_PARTS)
    chunkedList = [list(array) for array in chunkedArrays]
    
    # Normalisation of parameters
    differenceRuntime = runtimeNormalisation(dataset)
    differenceYear = yearNormalisation(dataset)
    
    # Cross validation
    numOfPart = 0
    if printAll:
        print('Number of parts: ' + str(NUM_OF_PARTS))
    overallSuccess = 0
    for part in chunkedList:
        
        # Define the base set
        baseSet = copy.deepcopy(titles)
        for element in dataset:
            if element in part:
                baseSet.remove(element)

        successInPart = 0
        
        # Analyze each part with the results of the others
        for movie in part:
            mostSim = mostSimilarMovies(movie, baseSet, dataset, ratings, weights, 
                                        NUM_OF_NEIGHBORS, differenceRuntime, differenceYear)

            # Check if similar movies are liked or not
            liked = 0
            shouldBeRecommended = False
            for i in range(len(mostSim)):
                if (ratings[mostSim[i]] == '5'):
                    liked += 1
            if (liked >= math.ceil(len(mostSim) / 2)):
                shouldBeRecommended = True
            
            # Calculate if solution is right
            succesful = checkSuccess(movie, ratings, shouldBeRecommended)
            if (succesful):
                successInPart += 1
            
        numOfPart += 1    
        successPercentage = (successInPart / len(part)) * 100
        overallSuccess += successPercentage
        if printAll:
            print("Success part " + str(numOfPart) + ': ' + str("{:.2f}".format(successPercentage)) + '%.')
    if printAll:
        print(f"Average success in all parts: {overallSuccess / NUM_OF_PARTS}%.")
    return overallSuccess/NUM_OF_PARTS



def readRatings(csvFile):
    
    dictionary = {}
    with open(csvFile, 'r', encoding="utf-8") as file:
        for line in file:
            if (line[0] == '2'):
                thisLine = line.split(',')
                title = thisLine[1]
                if (title[0] == '"'):
                    title = str(thisLine[1].split('"')[1] + ',' + thisLine[2].split('"')[0])
                    dictionary[title] = thisLine[5].split()[0]
                else:
                    dictionary[title] = thisLine[4].split()[0]
    
    return dictionary        


def readInformation(information):
    
    file = open(information)
    data = json.load(file)
    file.close()
    
    return data
    

# Returns the k most similar movies
def mostSimilarMovies(movie, baseSet, dataset, ratings, weights, k, differenceRuntime, differenceYear):
    
    # List with the similiraties between "movie" and each element of "baseSet"
    similarities = []
    
    # Calculate the similarity with each film
    for element in baseSet:
        similarities.append(similarity(movie, element, dataset, weights, differenceRuntime, differenceYear))
    
    # Search for the k minimum values
    posMin = np.array(similarities).argsort()[:k]
    mostSimilar = []
    for i in range(k):
        mostSimilar.append(baseSet[posMin[i]])
    
    return mostSimilar
    
    
# Returns the similarity between two films. Lower value = Higher similarity        
def similarity(movieA, movieB, dataSet, weights, differenceRuntime, differenceYear):
   
    movieAData = dataSet[movieA]
    movieBData = dataSet[movieB]
    totalScore = 0
    
    # Runtime
    runtimeScore = abs(movieAData['runtime'] - movieBData['runtime']) / differenceRuntime
    totalScore += runtimeScore * weights['runtime']
    
    # Language
    if (movieAData['original_language'] == movieBData['original_language']):
        languageScore = 0
    else:
        languageScore = 1
    totalScore += languageScore * weights['language']
    
    # Release year     
    releaseYearScore = abs(int(movieAData['release_date'].split('-')[0]) - 
                           int(movieBData['release_date'].split('-')[0])) / differenceYear
    totalScore += releaseYearScore * weights['release_year']
    
    # Genre
    genresA = movieAData['genres']
    genresB = movieBData['genres']
    if(len(genresA) != 0):
        coincidences = len(list(set(genresA).intersection(genresB)))
        genreScore = (len(genresA) - coincidences) / len(genresA)
        totalScore += genreScore * weights['genre']
    
    # Actors
    actorsA = movieAData['actors']
    actorsB = movieBData['actors']
    if(len(actorsA) != 0):
        coincidences = len(list(set(actorsA).intersection(actorsB)))
        actorsScore = (len(actorsA) - coincidences) / len(actorsA)
        totalScore += actorsScore * weights['actors']
    
    # Directors
    directorsA = movieAData['directors']
    directorsB = movieBData['directors']
    if(len(directorsA) != 0):
        coincidences = len(list(set(directorsA).intersection(directorsB)))
        directorsScore = (len(directorsA) - coincidences) / len(directorsA)
        totalScore += directorsScore * weights['directors']
    
    # Keywords
    keywordsA = movieAData['keywords']
    keywordsB = movieBData['keywords']
    if(len(keywordsA) != 0):
        coincidences = len(list(set(keywordsA).intersection(keywordsB)))
        keywordsScore = (len(keywordsA) - coincidences) / len(keywordsA)
        totalScore += keywordsScore * weights['keywords']
        
    return totalScore
     
 
# Returns the difference between the max and min runtimes    
def runtimeNormalisation(dataset):
       
    maxRuntime = 0
    minRuntime = 99999
    for movie in dataset:
        runtime = int(dataset[movie]['runtime']) 
        if (runtime > maxRuntime):
            maxRuntime = runtime
        if (runtime < minRuntime):
            minRuntime = runtime
            
    return maxRuntime - minRuntime
    

# Returns the difference between the max and min years
def yearNormalisation(dataset):
       
    maxYear = 0
    minYear = 99999
    for movie in dataset:
        release_date = dataset[movie]['release_date'] 
        year = int(release_date.split('-')[0])
        if (year > maxYear):
            maxYear = year
        if (year < minYear):
            minYear = year
            
    return maxYear - minYear
    

def checkSuccess(movie, ratings, shouldBeRecommended):
    success = False
    if (ratings[movie] == '5' and shouldBeRecommended == True):
        success = True
    elif (ratings[movie] == '1' and shouldBeRecommended == False):
        success = True    
    return success

if __name__ == "__main__":
    main('movie_features.json', 'ratings.csv')
    
    
    
    
    
    