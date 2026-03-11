import warnings
import csv

def gender_encoding(gender):
    """
    returns a single integer encoding for the user gender
    """
    if gender == 'M':
        return 1
    elif gender == 'F':
        return -1
    else:
        return 0

def unpack_data():
    """
    Unpacks data from ratings.csv,movies.csv and users.csv and combines them
    into 1 complete dataset csv file called data.csv

    filtering unnecessary or unused data in the process
    """
    try:
        ratings_data = []
        users_data = []
        movies_data = []

        combined_data = []

        with open('../data/ratings.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                ratings_data.append(row)

        with open('../data/users.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                users_data.append(row)

        with open('../data/movies.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                movies_data.append(row)

        genres = set([row[2] for row in movies_data])

        movie_index_map = {row[0]: i for i, row in enumerate(movies_data)}
        user_index_map = {row[0]: i for i, row in enumerate(users_data)}
        genre_index_map = {genre: index for index, genre in enumerate(genres)}

        for row in ratings_data:
            user_info = users_data[user_index_map[row[0]]]
            movie_info = movies_data[movie_index_map[row[1]]]
            rating = row[2]

            combined_row = [
                int(user_info[0]),
                gender_encoding(user_info[1]),
                int(user_info[2]),
                int(user_info[3]),
                int(movie_info[0]),
                genre_index_map[movie_info[2]],
                int(rating)
            ]

            combined_data.append(combined_row)

        with open('../data/data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId','gender','age','occupation','movieId','genre','rating'])
            writer.writerows(combined_data)

        return combined_data

    except FileNotFoundError:
        warnings.warn("File not found")

if __name__ == "__main__":
    unpacked_data = unpack_data()
