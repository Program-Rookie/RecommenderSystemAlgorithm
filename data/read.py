import pandas as pd
# ratings
rnames = ["UserID", "MovieID", "Rating", "Timestamp"]
ratings = pd.read_table("../data/ml-1m/ratings.dat", sep="::", names=rnames, engine='python')
print(ratings[:5])
# users
unames = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
users = pd.read_table("../data/ml-1m/users.dat", sep="::", names=unames, engine='python')
print(users[:5])
# movies
mnames = ["MovieID", "Title", "Genres"]
movies = pd.read_table("../data/ml-1m/movies.dat", sep="::", names=mnames, engine='python')
print(movies[:5])

data1 = pd.merge(ratings, users, on='UserID')
print(data1[:5])

data2 = pd.merge(ratings, movies, on='MovieID')
print(data2[:5])