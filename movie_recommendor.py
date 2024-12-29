#200101049 Yunus Emre Erkeleş
#200101108 Yalçın Özdemir
#200101016 Recep BAYKARA
#200101045 Muhammed Emin Sayan

import pandas as pd
import numpy as np
import ast
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, embedding_dim=128):
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_keywords = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)

        )
        self.scaler = StandardScaler()
        self.model = None
        self.movies_df = None
        self.feature_matrix = None
        self.embedding_dim = embedding_dim
        
    def clean_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        return ''
        
    def process_list_string(self, text):
        if isinstance(text, str):
            try:
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    return [str(item).strip() for item in items]
            except (ValueError, SyntaxError):
                text = text.strip('[]')
                items = [item.strip().strip("'\"") for item in text.split(',')]
                return [item for item in items if item]
        return []
        
    def preprocess_data(self, df):
        genres_binary = self.mlb_genres.fit_transform(df['Genre(s)'])
        genres_df = pd.DataFrame(genres_binary, columns=self.mlb_genres.classes_)
        
        keywords_binary = self.mlb_keywords.fit_transform(df['Keywords'])
        keywords_df = pd.DataFrame(keywords_binary, columns=self.mlb_keywords.classes_)
        
        cleaned_descriptions = df['Description'].fillna('').apply(self.clean_text)
        description_features = self.tfidf.fit_transform(cleaned_descriptions).toarray()
        
        rating_features = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean()).values.reshape(-1, 1)
        rating_features = self.scaler.fit_transform(rating_features)
        
        genres_weight = 1.0
        keywords_weight = 1.5
        description_weight = 2.0
        rating_weight = 0.5
        
        feature_matrix = np.hstack([
            genres_binary * genres_weight,
            keywords_binary * keywords_weight,
            description_features * description_weight,
            rating_features * rating_weight
        ])
        
        return self.scaler.fit_transform(feature_matrix)
    
    def build_model(self, input_dim):
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(self.embedding_dim, activation='relu'),
            BatchNormalization(),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(input_dim, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['mse']
        )
        return model
    
    def fit(self, df):
        self.movies_df = df.copy()
        self.feature_matrix = self.preprocess_data(df)
        
        self.model = self.build_model(self.feature_matrix.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            self.feature_matrix,
            self.feature_matrix,
            epochs=2,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
    def get_recommendations(self, movie_name, n_recommendations=10, similarity_threshold=0.3):
        if movie_name not in self.movies_df['Name'].values:
            raise ValueError(f"Movie '{movie_name}' not found in the database.")
            
        movie_idx = self.movies_df[self.movies_df['Name'] == movie_name].index[0]
        
        encoded_features = self.model.predict(self.feature_matrix)
        
        similarity_scores = cosine_similarity([encoded_features[movie_idx]], encoded_features)[0]
        
        input_movie_genres = set(self.movies_df.iloc[movie_idx]['Genre(s)'])
        
        genre_bonus = np.zeros_like(similarity_scores)
        for idx, movie_genres in enumerate(self.movies_df['Genre(s)']):
            if idx != movie_idx:
                common_genres = len(input_movie_genres.intersection(set(movie_genres)))
                genre_bonus[idx] = 0.1 * common_genres
        
        final_scores = similarity_scores + genre_bonus
        
        qualified_indices = np.where(final_scores >= similarity_threshold)[0]
        qualified_indices = qualified_indices[qualified_indices != movie_idx]
        
        sorted_indices = qualified_indices[np.argsort(final_scores[qualified_indices])[::-1]]
        
        top_indices = sorted_indices[:n_recommendations]
        recommendations = self.movies_df.iloc[top_indices][['Name', 'Description', 'Genre(s)', 'IMDB_Rating']]
        
        recommendations['Similarity_Score'] = final_scores[top_indices]
        
        return recommendations.sort_values('Similarity_Score', ascending=False)

def main():
    df = pd.read_csv("clean_data\latest5000_5_clean_imdb_data.csv")
    
    recommender = MovieRecommender(embedding_dim=128)
    recommender.fit(df)
    
    movie_name = "Pirates of the Caribbean: The Curse of the Black Pearl"
    recommendations = recommender.get_recommendations(movie_name, n_recommendations=10)
    
    print(f"\nTop 10 recommendations for {movie_name}:")
    for idx, row in recommendations.iterrows():
        print(f"\nMovie: {row['Name']}")
        print(f"IMDB Rating: {row['IMDB_Rating']}")
        print(f"Similarity Score: {row['Similarity_Score']:.3f}")
        print(f"Description: {row['Description'][:200]}...")

if __name__ == "__main__":
    main()
