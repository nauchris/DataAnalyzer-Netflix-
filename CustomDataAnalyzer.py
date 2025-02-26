"""
Netflix Data Analyzer Script
This script loads a Netflix dataset, cleans and preprocesses it, analyzes trends
such as genre popularity, top-rated movies, and content release counts by year,
and visualizes these insights using Pandas and Matplotlib.
"""
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Load the Netflix dataset from a CSV file into a pandas DataFrame.
    :param filepath: Path to the CSV data file.
    :return: DataFrame containing the Netflix dataset.
    """
    df = pd.read_csv(filepath)
    # Optionally, handle API or other data source if provided (not implemented here).
    return df

def clean_data(df):
    """
    Clean and preprocess the Netflix dataset:
    - Drop duplicates
    - Strip whitespace from string fields
    - Handle missing values (e.g., drop if title is missing)
    - Ensure correct data types (e.g., convert release_year to numeric)
    :param df: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    # Drop duplicate entries, if any
    df = df.drop_duplicates()
    
    # Trim whitespace from string columns to tidy up data
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    
    # Drop rows with missing title (assuming title is essential)
    if 'title' in df.columns:
        df = df[df['title'].notna()]
    
    # Convert release_year to numeric (if not already)
    if 'release_year' in df.columns:
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    return df

def summary_stats(df):
    """
    Calculate summary statistics for the dataset.
    :param df: DataFrame.
    :return: Dict of summary statistics (total titles, count of movies vs shows, year range, average rating if available).
    """
    stats = {}
    stats['total_titles'] = len(df)
    # Count movies vs TV shows if 'type' column exists
    if 'type' in df.columns:
        stats['num_movies'] = int((df['type'] == 'Movie').sum())
        stats['num_tv_shows'] = int((df['type'] == 'TV Show').sum())
    # Year range of content
    if 'release_year' in df.columns:
        stats['earliest_year'] = int(df['release_year'].min())
        stats['latest_year'] = int(df['release_year'].max())
    # Average IMDb score if available
    for col in ['IMDb Score', 'IMDB Score', 'imdb_score']:
        if col in df.columns:
            stats['average_imdb_score'] = float(df[col].mean())
            break
    return stats

def get_top_genres(df, n=10):
    """
    Get the top N genres by number of titles.
    Assumes a genre list column (like 'listed_in') with comma-separated genres.
    :param df: DataFrame.
    :param n: Number of top genres to return.
    :return: Series of genre counts for the top N genres.
    """
    genre_col = 'listed_in' if 'listed_in' in df.columns else ('genre' if 'genre' in df.columns else None)
    if genre_col:
        # Split the genre strings and explode into separate rows for counting
        genre_series = df[genre_col].str.split(',').explode().str.strip()
        genre_counts = genre_series.value_counts()
        return genre_counts.head(n)
    else:
        return pd.Series(dtype=int)  # return empty Series if no genre column

def get_top_rated_movies(df, n=10):
    """
    Get the top N highest-rated movies.
    Assumes there's a numeric rating column (e.g., 'IMDb Score') and a 'type' column to filter movies.
    :param df: DataFrame.
    :param n: Number of top movies to return.
    :return: DataFrame of top N movies sorted by rating.
    """
    # Identify the rating column to use
    rating_col = None
    for col in ['IMDb Score', 'IMDB Score', 'imdb_score', 'rating']:
        if col in df.columns:
            # Skip 'rating' if it's non-numeric (like content rating)
            if col == 'rating' and df[col].dtype == 'object':
                continue
            rating_col = col
            break
    if rating_col:
        movies_df = df[df['type'] == 'Movie'] if 'type' in df.columns else df
        top_movies = movies_df.sort_values(by=rating_col, ascending=False).head(n)
        return top_movies
    else:
        return pd.DataFrame()  # if no suitable rating column found

def get_year_counts(df):
    """
    Get the number of titles released each year.
    :param df: DataFrame.
    :return: Series indexed by release year, with counts of titles for each year.
    """
    if 'release_year' in df.columns:
        return df['release_year'].value_counts().sort_index()
    else:
        return pd.Series(dtype=int)

def plot_genre_distribution(top_genres):
    """
    Plot a bar chart for genre distribution.
    :param top_genres: Series of genre counts (for top genres).
    """
    plt.figure(figsize=(8,4))
    plt.bar(top_genres.index, top_genres.values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Number of Titles')
    plt.title(f'Top {len(top_genres)} Genres on Netflix')
    plt.tight_layout()
    # Normally plt.show() would be called here to display the plot.

def plot_top_rated_movies(top_movies):
    """
    Plot a horizontal bar chart for top-rated movies.
    :param top_movies: DataFrame of top movies with their ratings.
    """
    if top_movies.empty:
        return
    # Determine which rating column was used
    rating_col = None
    for col in ['IMDb Score', 'IMDB Score', 'imdb_score', 'rating']:
        if col in top_movies.columns:
            if col == 'rating' and top_movies[col].dtype == 'object':
                continue
            rating_col = col
            break
    titles = top_movies['title'] if 'title' in top_movies.columns else top_movies.index.astype(str)
    plt.figure(figsize=(8,4))
    plt.barh(titles, top_movies[rating_col], color='orange')
    plt.xlabel('Rating')
    plt.title(f'Top {len(top_movies)} Rated Movies')
    plt.gca().invert_yaxis()  # highest rating at top
    plt.tight_layout()
    # Normally plt.show() would be called to display the plot.

def plot_yearly_content(year_counts):
    """
    Plot a line chart for number of titles released each year.
    :param year_counts: Series with index as year and values as count of titles.
    """
    if year_counts.empty:
        return
    plt.figure(figsize=(8,4))
    plt.plot(year_counts.index, year_counts.values, marker='o', color='green')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')
    plt.title('Netflix Content Count by Year')
    plt.tight_layout()
    # Normally plt.show() would be called to display the plot.

def main():
    """
    Main function to execute the analysis.
    """
    # Load data from CSV
    data_path = 'netflix_titles.csv'  # Update this path to your dataset file
    df = load_data(data_path)
    
    # Clean the data
    df = clean_data(df)
    
    # Summary statistics
    stats = summary_stats(df)
    print("Summary Statistics:")
    print(f"- Total titles: {stats.get('total_titles', 0)}")
    if 'num_movies' in stats:
        print(f"- Movies: {stats['num_movies']} | TV Shows: {stats['num_tv_shows']}")
    if 'earliest_year' in stats and 'latest_year' in stats:
        print(f"- Year range: {stats['earliest_year']} to {stats['latest_year']}")
    if 'average_imdb_score' in stats:
        print(f"- Average IMDb Score: {stats['average_imdb_score']:.2f}")
    print("")
    
    # Top genres
    top_genres = get_top_genres(df, n=10)
    print("Top 10 Genres by Number of Titles:")
    for genre, count in top_genres.items():
        print(f"{genre}: {count}")
    print("")
    
    # Top-rated movies
    top_movies = get_top_rated_movies(df, n=10)
    if not top_movies.empty:
        print("Top 10 Rated Movies:")
        for idx, row in top_movies.iterrows():
            title = row['title'] if 'title' in row else str(idx)
            # Determine rating for output
            rating_val = None
            for col in ['IMDb Score', 'IMDB Score', 'imdb_score', 'rating']:
                if col in row.index:
                    if col == 'rating' and top_movies[col].dtype == 'object':
                        continue
                    rating_val = row[col]
                    break
            print(f"{title} (Rating: {rating_val})")
    else:
        print("No numeric rating column found in dataset to identify top-rated movies.")
    print("")
    
    # Yearly content distribution
    year_counts = get_year_counts(df)
    if not year_counts.empty:
        max_year = int(year_counts.idxmax())
        print(f"Year with most releases: {max_year} ({year_counts.max()} titles)")
    print("")
    
    # Generate visualizations (not displayed here)
    plot_genre_distribution(top_genres)
    plot_top_rated_movies(top_movies)
    plot_yearly_content(year_counts)
    print("Visualizations generated (not displayed in this environment).")

if __name__ == "__main__":
    main()
