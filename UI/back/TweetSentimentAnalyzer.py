import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import numpy as np 
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import io
import base64
from collections import Counter

class TweetSentimentAnalyzer:
    def __init__(self):
        self.languages_dict = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'Deutsch'}
        self.model_names = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment"
            
            ]
        

    def open_file(self, year, language, OG, analyzed):
            # Get the parent directory of the current working directory
            # Obtient le répertoire courant
            current_dir = os.getcwd()

            # Remonte de deux niveaux pour arriver à 'FoodSafetyEngagement'
            parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
            if analyzed:
                data_dir = os.path.join(parent_dir, "Data", "TwitterData", "AnalyzedDatasets")
            else:
                data_dir = os.path.join(parent_dir, "Data", "TwitterData", "RawDatasets")
            
            filename = f"tweets_data_{language}_{year}"
            if analyzed:
                filename += "_analyzed"
            else:
                if OG:
                    filename += "_VO"
                elif OG == False:
                    filename += "_VE"
            
            file_path = os.path.join(data_dir, filename + '.csv')
            tweets_data = pd.read_csv(file_path, sep=',')
            print(file_path)
            
            if language != 'en' and OG == True:
                if 'content_translated' not in tweets_data.columns:
                    print(f"Translating {len(tweets_data['content'])} tweets from {language.upper()} to English... for file {filename}")
                    tqdm.pandas(desc="Translating tweets")
                    
                    # Ensure the column is created before starting translation
                    tweets_data['content_translated'] = tweets_data['content'].apply(lambda text: "")

                    # Apply translation
                    tweets_data['content_translated'] = tweets_data['content'].progress_apply(
                        lambda text: self.translate_sentence(text, language)
                    )
                    print(f"Translated {len(tweets_data['content'])} tweets from {language.upper()} to English.")
            
            try:
                tweets_data.to_csv(file_path, mode='w', index=False, encoding='utf-8')
                return tweets_data
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
        
        
    def translate_sentence(self, text, source_language):
            try:
                # Check if the input is a string
                if not isinstance(text, str):
                    return text
                
                # Load the model and tokenizer for translation
                model_name = f"Helsinki-NLP/opus-mt-{source_language}-en"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                # Preprocess the text for the model
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

                # Perform the translation
                outputs = model.generate(**inputs, max_length=512)

                # Decode and return the translated text
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated_text

            except Exception as e:
                return f"Error during translation: {e}"

        
        
    def analyze_sentiment_ai(self, tweet, tokenizer, model):
        if not isinstance(tweet, str):
            return None  # or some default sentiment score, e.g., [0, 1, 0] for neutral

        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits[0].detach().numpy()
        scores = softmax(scores)
        return scores

        

    def sentiment_analyzing(self, year, language):
        tqdm.pandas()
        model_names = self.model_names

        parent_dir = os.path.dirname(os.getcwd())
        data_dir = os.path.join(parent_dir, "Data", "TwitterData", "AnalyzedDatasets")

        num_csv_files = 0

        # Load tweet data based on language
        if language == 'en':
            tweets_data = self.open_file(year, language, None, False)
        else:
            try:
                tweets_data = pd.concat([
                    self.open_file(year, language, True, False),
                    self.open_file(year, language, False, False)
                ], ignore_index=True)
            except:
                tweets_data = self.open_file(year, language, True, False)
            
        print(tweets_data.head())
        for model_name in model_names:
            print(f"Analyzing sentiment for {year} in {language} using model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            ai_sentiments = tweets_data['content_translated' if 'content_translated' in tweets_data.columns else 'content'].progress_apply(lambda tweet: self.analyze_sentiment_ai(tweet, tokenizer, model))

            # Perform sentiment analysis
            tqdm.pandas(desc=f"Analyzing sentiment with {model_name}")

            # Create columns for this specific model
            model_id = model_name.split("/")[-1]
            tweets_data[f'{model_id}_positive_score'] = ai_sentiments.apply(lambda x: x[2])
            tweets_data[f'{model_id}_negative_score'] = ai_sentiments.apply(lambda x: x[0])
            tweets_data[f'{model_id}_neutral_score'] = ai_sentiments.apply(lambda x: x[1])
            tweets_data[f'{model_id}_sentiment'] = ai_sentiments.apply(lambda x: 2 if x[2] == max(x) else (0 if x[0] == max(x) else 1))

        # Create the file name
        file_name = f'tweets_data_{language}_{year}_analyzed'
        file_path = os.path.join(data_dir, file_name + '.csv')
        tweets_data.to_csv(file_path, mode='w', index=False, encoding='utf-8')

        num_csv_files += 1
            
        return model_names




    def save_to_csv(self, dataframe, file_path):
        if os.path.exists(file_path):
            dataframe.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8')
        else:
            dataframe.to_csv(file_path, mode='w', index=False, encoding='utf-8')


    def wordcloud(self, year, language):
        def clean_word(word):
            """Nettoyer les mots en retirant la ponctuation."""
            return re.sub(r'[^\w\s]', '', word).lower()

        # Mapping from short code to full language name for stopwords
        lang_dict = {
            'fr': 'french',
            'es': 'spanish',
            'en': 'english',
            'de': 'german'
        }

        # Mots vides personnalisés
        stop_words_personal = '''food waste climate change climat climatechange https http com www twitter pic status co amp www youtube foodwaste
        climatique changement climatique alimentaire déchets alimentaires comida residuos cambio climático alimentación'''
        stop_words_personal_set = set(stop_words_personal.split())

        # Charger les données pour l'année spécifiée
        if language == 'en':
            tweets_data = self.open_file(year, language, None, False)
        else:
            tweets_data = pd.concat([
                self.open_file(year, language, True, False),
                self.open_file(year, language, False, False)
            ])

        # Déterminer les mots vides pour la langue spécifiée
        stop_words_language = set(stopwords.words(lang_dict[language]))
      

        # Fusionner tous les ensembles de mots vides pour obtenir un ensemble unique
        stop_words = stop_words_personal_set.union(stopwords.words('english')).union(stop_words_language)

        # Dictionnaire pour stocker la fréquence des mots
        word_freq = {}

        # Boucle sur chaque tweet pour calculer la fréquence des mots
        for index, row in tweets_data.iterrows():
            tweet = row['content']
            words = tweet.split()
            for word in words:
                word = clean_word(word)
                if word not in stop_words and word != '':
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Créer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Créer la figure et l'axe pour afficher le nuage de mots
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"Word Cloud for Year {year} in language {language} with {len(tweets_data)} posts")
        ax.axis('off')

        # Encode the plot as a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(img.getvalue()).decode()
        
        return graph_base64




    def CircularGraph(self, years, language):
        num_years = len(years)
        cols = 2
        rows = num_years  # 1 row per year, 2 cols for VADER and AI graphs

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))  # Adjust size for better visibility
        axes = axes.flatten()  # Flatten in case we have more than 1 row

        for i, year in enumerate(years):
            try:
                tweets_data_analyzed = self.open_file(year, language, None, True)
                print(f"Sentiment data for {year} in {language} already exists.")
            except:
                self.sentiment_analyzing(year, language)
                tweets_data_analyzed = self.open_file(year, language, None, True)

            model_names = self.model_names
            labels = ['Positive', 'Negative', 'Neutral']
            colors = ['#DEB7BC', '#75343C', '#A3A3A3']
            explode = (0.1, 0, 0)  # explode 1st slice

            for j, model_name in enumerate(model_names):
                model_id = model_name.split("/")[-1]
                try:
                    positive_count_ai = (tweets_data_analyzed[f'{model_id}_sentiment'] == 2).sum()
                    negative_count_ai = (tweets_data_analyzed[f'{model_id}_sentiment'] == 0).sum()
                    neutral_count_ai = (tweets_data_analyzed[f'{model_id}_sentiment'] == 1).sum()
                except KeyError as e:
                    print(f"KeyError: {e} not found in the data.")
                    continue

                sizes_ai = [positive_count_ai, negative_count_ai, neutral_count_ai]
                ax_index = cols * i + j  # Correcting the index
                wedges, texts, autotexts = axes[ax_index].pie(sizes_ai, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
                
                for text in texts + autotexts:  # Reduce text size
                    text.set_fontsize(8)
                    
                axes[ax_index].set_title(f'AI Sentiment Analysis for {year}\nin {language} on {len(tweets_data_analyzed)} Tweets using {model_id}', fontsize=10)

        plt.tight_layout(pad=2.0)  # Adjust the padding between and around subplots
        plt.show()


        
    def BarGraph(self, years, language):
        num_years = len(years)
        num_models = len(self.model_names)
        
        avg_positive_scores = {model_name: [] for model_name in self.model_names}
        avg_negative_scores = {model_name: [] for model_name in self.model_names}
        avg_neutral_scores = {model_name: [] for model_name in self.model_names}
        year_labels = []

        for year in years:
            try:
                tweets_data_analyzed = self.open_file(year, language, None, True)
                print(f"Sentiment data for {year} in {language} already exists.")
            except:
                self.sentiment_analyzing(year, language)
                tweets_data_analyzed = self.open_file(year, language, None, True)

            for model_name in self.model_names:
                model_id = model_name.split("/")[-1]
                try:
                    avg_positive_scores[model_name].append((tweets_data_analyzed[f'{model_id}_sentiment'] == 2).mean())
                    avg_negative_scores[model_name].append((tweets_data_analyzed[f'{model_id}_sentiment'] == 0).mean())
                    avg_neutral_scores[model_name].append((tweets_data_analyzed[f'{model_id}_sentiment'] == 1).mean())
                except KeyError as e:
                    print(f"KeyError: {e} not found in the data.")
                    avg_positive_scores[model_name].append(0)
                    avg_negative_scores[model_name].append(0)
                    avg_neutral_scores[model_name].append(0)
            
            year_labels.append(year)

        bar_width = 0.2
        bar_positions = np.arange(num_years)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model_name in enumerate(self.model_names):
            r1 = bar_positions + i * bar_width
            ax.bar(r1, avg_positive_scores[model_name], color='#76C7C0', width=bar_width, edgecolor='grey', label=f'Positive - {model_name.split("/")[-1]}')
            ax.bar(r1, avg_negative_scores[model_name], color='#D9534F', width=bar_width, edgecolor='grey', bottom=avg_positive_scores[model_name], label=f'Negative - {model_name.split("/")[-1]}')
            ax.bar(r1, avg_neutral_scores[model_name], color='#F0AD4E', width=bar_width, edgecolor='grey', bottom=[sum(x) for x in zip(avg_positive_scores[model_name], avg_negative_scores[model_name])], label=f'Neutral - {model_name.split("/")[-1]}')

        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Average Sentiment Score', fontweight='bold')
        ax.set_title(f'Average Sentiment Scores per Year in {self.languages_dict[language]}', fontweight='bold')
        ax.set_xticks(bar_positions + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(year_labels)
        ax.legend()

        plt.tight_layout()
        plt.show()

        
    def BarGraphYearLanguage(self,years, languages):
    # Example loading tweet data by language for the given year (assuming AnalyzeTwitter functions are defined elsewhere)
        model_names = self.model_names
        model_name = model_names[0]  # Selecting the first model for example
        model_name_segment = model_name.split("/")[-1]
        sentiment_column = f"{model_name_segment}_sentiment"

        all_data = []

        for year in years:
            for language in languages:
                # Load tweet data for the given year and language
                tweets_data = self.open_file(year, language, None, True)
                tweets_data['language'] = language
                tweets_data['year'] = year
                all_data.append(tweets_data)

        # Combine all DataFrames into one
        tweets_data_combined = pd.concat(all_data, ignore_index=True)

        # Calculate average sentiment score per language and year
        sentiment_by_language_year = tweets_data_combined.groupby(['year', 'language'])[sentiment_column].mean().unstack()

        # Plotting the sentiment scores as a bar graph
        sentiment_by_language_year.plot(kind='bar', figsize=(12, 8))
        plt.xlabel('Year')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment Scores by Language and Year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title='Language')
        plt.show()
        


    def SentimentEvolution(self, rangeYear, language):
        # Exemple de chargement des données de tweets pour les années données et la langue
        model_names = self.model_names
        model_name = model_names[0]  # Sélectionner le premier modèle pour l'exemple
        model_name_segment = model_name.split("/")[-1]
        sentiment_column = f"{model_name_segment}_sentiment"

        all_data = []

        start_year, end_year = rangeYear

        # Utiliser range avec ces variables
        for year in range(start_year, end_year + 1):
            # Charger les données de tweets pour l'année et la langue données
            tweets_data = self.open_file(year, language, None, True)
            tweets_data['language'] = language
            tweets_data['year'] = year
            all_data.append(tweets_data)

        # Combiner tous les DataFrames en un seul
        tweets_data_combined = pd.concat(all_data, ignore_index=True)

        # Calculer la moyenne des scores de sentiment par année
        sentiment_by_year = tweets_data_combined.groupby('year')[sentiment_column].mean().reset_index()

        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sentiment_by_year['year'], sentiment_by_year[sentiment_column], marker='o', linestyle='-')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Sentiment Score')
        ax.set_title(f'Average Sentiment Scores for {language.upper()} Tweets ({rangeYear})')
        ax.grid(True)
        plt.tight_layout()

        # Encoder le graphique comme une chaîne base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(img.getvalue()).decode()

        return graph_base64

    def plot_sentiment_map(self, year, country):
        # Load tweet data for the specified year and country
        tweets_data = self.AnalyzeTwitter.open_file(year, country, None, True)
        tweets_data['country'] = country
        
        # Add a 'region' column based on country
        country_to_regions = {
            'en': ['United States of America', 'United Kingdom', 'India', 'Canada', 'Australia'],
            'fr': ['France', 'Belgium', 'Switzerland', 'Canada']
        }
        
        regions = country_to_regions.get(country, [country])
        
        extended_tweets_data = []
        for index, row in tweets_data.iterrows():
            for region in regions:
                new_row = row.copy()
                new_row['region'] = region
                extended_tweets_data.append(new_row)
        
        tweets_data_extended = pd.DataFrame(extended_tweets_data)
        
        # Calculate average sentiment score per region
        region_sentiment = tweets_data_extended.groupby('region')[self.sentiment_column].mean().reset_index()
        
        # Load world shapefile
        world_shp_path = "C:/Users/theoh/Documents/FoodSafetyEngagement/AI_Models_NLP/ne_110m_admin_0_countries.shp"  # Replace with your local path to the .shp file
        world = gpd.read_file(world_shp_path)
        
        # Add a sentiment column to the world map
        world[self.sentiment_column] = -1  # Initialize column with a default value
        
        # Add sentiment values from region_sentiment to world
        for index, row in region_sentiment.iterrows():
            world.loc[world['NAME'] == row['region'], self.sentiment_column] = row[self.sentiment_column]
        
        # Define sentiment label intervals
        def map_sentiment_label(score):
            if score == -1:
                return 'No Data'
            elif 0 <= score < 1:
                return 'Negative'
            elif 1 <= score < 1.5:
                return 'Neutral'
            elif score >= 1.5:
                return 'Positive'
        
        # Apply mapping to sentiment_column
        world['sentiment_label'] = world[self.sentiment_column].apply(map_sentiment_label)
        
        # Print counts of each sentiment
        print("Number of regions with 'Neutral' sentiment:", (world['sentiment_label'] == 'Neutral').sum())
        print("Number of regions with 'Negative' sentiment:", (world['sentiment_label'] == 'Negative').sum())
        print("Number of regions with 'Positive' sentiment:", (world['sentiment_label'] == 'Positive').sum())
        print("Number of regions with no data:", (world['sentiment_label'] == 'No Data').sum())
        
        # Check if sentiment_column contains values
        if world[self.sentiment_column].empty:
            raise ValueError("The sentiment_column is empty after joining. Check your data.")
        
        # Normalize sentiment_column values for a continuous colormap
        norm = Normalize(vmin=world[self.sentiment_column].min(), vmax=world[self.sentiment_column].max())
        mapper = ScalarMappable(norm=norm, cmap='viridis')
        
        # Plot the map with sentiment_column as color
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.plot(column=self.sentiment_column, ax=ax, cmap='viridis', linewidth=0.8, edgecolor='0.8')
        
        # Add a colorbar
        fig.colorbar(mapper, ax=ax, orientation='vertical', label='Sentiment Score')
        
        # Add labels and a title
        ax.set_title(f"World Map with Average Sentiment Scores ({year})")
        plt.show()

    def CountWordSentiment(self, year, language):
        model_names = self.model_names
        try:
            tweets_data = self.open_file(year, language, None,True)
            print(f"Sentiment data for {year} already exists.")
        except Exception as e:
            print(f"Error opening file for {year}: {e}")
            self.sentiment_analyzing(year)
            tweets_data = self.open_file(year, language, None,True)


        stop_words = set(stopwords.words('english'))
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#DEB7BC', '#75343C', '#A3A3A3']

        for model_name in model_names:
            model_id = model_name.split("/")[-1]
            try:
                try:
                    content = tweets_data['content_translated']
                except:
                    content = tweets_data['content']
                sentiments = tweets_data[f'{model_id}_sentiment']
            except KeyError as e:
                print(f"KeyError: {e} not found in the data.")
                continue
            
            # Clean and tokenize the content, removing stopwords
            cleaned_content = []
            for text in content:
                words = re.findall(r'\b\w+\b', text.lower())
                filtered_words = [word for word in words if word not in stop_words]
                cleaned_content.extend(filtered_words)
            
            word_counter = Counter(cleaned_content)
            most_common_words = word_counter.most_common(10)
            words = [word for word, _ in most_common_words]

            positive_counts = []
            negative_counts = []
            neutral_counts = []

            for word in words:
                word_data = tweets_data[content.str.contains(r'\b' + re.escape(word) + r'\b', regex=True)]
                positive_counts.append((word_data[f'{model_id}_sentiment'] == 2).sum())
                negative_counts.append((word_data[f'{model_id}_sentiment'] == 0).sum())
                neutral_counts.append((word_data[f'{model_id}_sentiment'] == 1).sum())

            fig, ax = plt.subplots()

            bar_width = 0.5
            bar_positions = range(len(words))
            
            ax.barh(bar_positions, positive_counts, bar_width, color='#DEB7BC', label='Positive')
            ax.barh(bar_positions, negative_counts, bar_width, left=positive_counts, color='#75343C', label='Negative')
            ax.barh(bar_positions, neutral_counts, bar_width, left=[i + j for i, j in zip(positive_counts, negative_counts)], color='#A3A3A3', label='Neutral')
            
            ax.set_yticks(bar_positions)
            ax.set_yticklabels(words)
            ax.set_xlabel('Count')
            ax.set_title(f'Top 10 Words by Sentiment for Model: {model_id} ({year})')
            ax.legend()

            # Encode the plot as a base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

            graph_base64 = base64.b64encode(img.getvalue()).decode()
            
        return graph_base64

if __name__ == "__main__":
    analyzer = TweetSentimentAnalyzer()
    
