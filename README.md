# Movie-Recommendation-Program

Final project for "Intelligent Systems" course. Here we have a system that, starting from a list of personal ratings and applying Case-based Reasoning (CBR), is able to recommend users new films based on multiple weighted features. These features are: runtime, release year, actors, directors, genre, keywords and language. Weights were not randomly chosen, but using a genetic algorithm which searched for best accuracy. The model uses K-fold cross validation in order to check its correct functioning.

The file movie_features.json contains all the required information about the movies.

The file ratings.csv contains examples of personal ratings about 200 different movies.
