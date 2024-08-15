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
import glob
from tqdm import tqdm
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-GUI
from collections import Counter
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import io
import base64

# Reste de votre code

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import concurrent.futures
import io
import base64

import matplotlib
print("Current backend:", matplotlib.get_backend())

# Configure the backend if necessary
matplotlib.use('Agg')  # Remplacez 'Agg' par un autre backend si besoin

class RedditSentimentAnalyzerBack:
    def __init__(self):
        self.languages_dict = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'Deutsch'}
        self.country_language_map = {
            'france': 'fr',         # Français
            'germany': 'de',        # Allemand
            'usa': 'en',            # Anglais
            'canada': 'en',         # Anglais
            'mexico': 'es',         # Espagnol
            'denmark': 'da',        # Danois
            'sweden': 'sv',         # Suédois
            'china': 'zh',          # Chinois (Mandarin)
            'netherlands': 'nl',    # Néerlandais
            'india': 'hi',          # Hindi
            'italy': 'it'           # Italien
        }


        self.model_names = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            ]




#------------------------------------------------Data Functions------------------------------------------------------------

    def open_file_country(self, start_year, end_year, start_month, end_month, country, theme, analyzed):
        print('onest dans openfilecountry')
        # Obtient le répertoire courant
        current_dir = os.getcwd()
        print('start_year type',type(start_year))
        print('end_year type',type(end_year))
        print('start_month type',start_month)
        print('end_month type',type(end_month))
        print('country type',type(country))
        # Remonte de deux niveaux pour arriver à 'FoodSafetyEngagement'
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

        if analyzed:
            print('on est dans analyze')
            data_dir = os.path.join(parent_dir, "Data", "RedditData", "AnalyzedDatasets", theme, country)
            for year in range(int(start_year), int(end_year) + 1):
                filename = f"{theme}_data_{country}_{str(year)}_analyzed.csv"
                file_path = os.path.join(data_dir, filename)

                # Vérifiez si le fichier existe
                if not os.path.exists(file_path): 
                    print('Analyzing sentiment for', year)
                    self.sentiment_analyzing(year, country, theme)
                else:
                    print(f'File for {year} already exists.')

        else:
            data_dir = os.path.join(parent_dir, "Data", "RedditData", "FilteredDatasets", theme, country)

        # Initialise une liste pour stocker les DataFrames
        reddit_data_list = []

        # Boucle à travers chaque année entre start_year et end_year inclus
        for year in range(int(start_year), int(end_year) + 1):

            if analyzed:
                filename = f"{theme}_data_{country}_{str(year)}_analyzed.csv"
            else:
                filename = f"{theme}_data_{country}_{str(year)}.csv"
            print(filename)
            file_path = os.path.join(data_dir, filename)
            print(file_path)
            
            # Vérifie si le fichier existe avant de le lire
            if os.path.isfile(file_path):
                # Lire le fichier CSV
                reddit_data_year = pd.read_csv(file_path, header=0)
                # Ajouter une colonne 'year' avec l'année correspondante
                reddit_data_year['year'] = str(year)
                
                # Convertir la colonne 'created' en datetime
                reddit_data_year['created'] = pd.to_datetime(reddit_data_year['created'])

                # Ajouter le DataFrame à la liste
                reddit_data_list.append(reddit_data_year)

        # Combiner tous les DataFrames en un seul
        reddit_data = pd.concat(reddit_data_list, ignore_index=True)

        # Créer les dates de début et de fin pour le filtrage
        start_date = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
        end_date = pd.Timestamp(year=int(end_year), month=int(end_month), 
                                day=pd.Timestamp(year=int(end_year), month=int(end_month), day=1).days_in_month)

        # Filtrer les données en fonction de la plage de dates
        filtered_data = reddit_data[(reddit_data['created'] >= start_date) & (reddit_data['created'] <= end_date)]
        

        return filtered_data
    
    

    def translate_sentence(self, text, source_lang):
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-en'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs.input_ids.clone()

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs)
            translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            print(f"Translation failed: {e}")
            return None

    def translate_content(self, file_path):
        file_path_for_translation = file_path
        

        reddit_data = pd.read_csv(file_path_for_translation, header=0)

        reddit_data['content'] = reddit_data.apply(
            lambda row: f"{row['title']} {row['body']}" if pd.notnull(row['title']) and pd.notnull(row['body']) else (
                row['title'] if pd.notnull(row['title']) else row['body']), axis=1
        )

        country = file_path.split('_')[-1].split('.')[0]
        language = self.country_language_map.get(country.lower(), 'en')

        if 'content_translated' in reddit_data.columns and not reddit_data['content_translated'].isnull().all():
            print("Content already translated. No need to translate again.")
        elif language == 'en':
            print(f"No translation needed for {country} data in {language.upper()}.")
            reddit_data['content_translated'] = reddit_data['content']
        else:
            print(f"Translating {len(reddit_data['content'])} posts from {language.upper()} to English for file {file_path_for_translation}")
            tqdm.pandas(desc="Translating posts")

            if 'content_translated' not in reddit_data.columns:
                reddit_data['content_translated'] = ""

            try:
                reddit_data['content_translated'] = reddit_data['content'].progress_apply(
                    lambda text: self.translate_sentence(text, language)
                )
                print(f"Translated {len(reddit_data['content'])} posts from {language.upper()} to English.")
            except Exception as e:
                print(f"Translation failed: {e}")
                pass

        # Sauvegarde les colonnes nécessaires dans le fichier CSV
        reddit_data.to_csv(file_path_for_translation, mode='w', index=False, encoding='utf-8')
        return file_path_for_translation
    
    def analyze_sentiment_ai(self, tweet, tokenizer, model):
            # Your sentiment analysis logic here
            inputs = tokenizer(tweet, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            return predicted_class, probs.tolist()


    def analyze_sentiment(self, file_path, model_names):
        reddit_data = pd.read_csv(file_path)
        
        analyzed_file_path = file_path.replace('FilteredDatasets', 'AnalyzedDatasets').replace('.csv', '_analyzed.csv')
        
        if not os.path.exists(analyzed_file_path):
            os.makedirs(os.path.dirname(analyzed_file_path), exist_ok=True)
        
        if os.path.exists(analyzed_file_path):
            print('Le ichier existe effectivement on check sil est fullfill')
            analyzed_data = pd.read_csv(analyzed_file_path)
            incomplete_models = []
            for model_name in model_names:
                model_id = model_name.split("/")[-1]
                if (f'{model_id}_sentiment' in analyzed_data.columns and
                    analyzed_data[f'{model_id}_sentiment'].count() == len(reddit_data)):
                    print(f"Analyzed file already exists and is complete for model {model_name}: {analyzed_file_path}")
                else:
                    incomplete_models.append(model_name)
                    print(f"Analyzed file exists but is incomplete for model {model_name}. Resuming analysis...")
            reddit_data = analyzed_data
            model_names = incomplete_models  # Update model_names to only include incomplete models
        
        def analyze_tweet_sentiment(tweet, model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return self.analyze_sentiment_ai(tweet, tokenizer, model)
        
        for model_name in model_names:
            model_id = model_name.split("/")[-1]
            tqdm.write(f"Analyzing sentiment for {file_path} using model {model_name}...")
            ai_sentiments = []

            if reddit_data['content_translated'].str.strip().empty:
                tqdm.write('No content to analyze')
                continue 
            
            for i, tweet in tqdm(enumerate(reddit_data['content_translated']), desc="Analyzing tweets", total=len(reddit_data['content_translated'])):
                if f'{model_id}_sentiment' in reddit_data.columns and not pd.isna(reddit_data.loc[i, f'{model_id}_sentiment']):
                    continue  # Skip already analyzed tweets
                sentiment = analyze_tweet_sentiment(tweet, model_name)
                if sentiment is not None:
                    ai_sentiments.append((i, sentiment))
                    reddit_data.loc[i, f'{model_id}_positive_score'] = sentiment[1][0][2]
                    reddit_data.loc[i, f'{model_id}_negative_score'] = sentiment[1][0][0]
                    reddit_data.loc[i, f'{model_id}_neutral_score'] = sentiment[1][0][1]
                    reddit_data.loc[i, f'{model_id}_sentiment'] = sentiment[0]

                    # Sauvegarde après chaque mise à jour de sentiment
                    reddit_data.to_csv(analyzed_file_path, mode='w', index=False, encoding='utf-8')

        return analyzed_file_path
    
    
    def sentiment_analyzing(self, year,country,theme):
        # Obtient le répertoire courant 
        current_dir = os.getcwd()

        # Remonte de deux niveaux pour arriver à 'FoodSafetyEngagement'
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

        # Ajoute les chemins nécessaires pour atteindre 'Data'
        data_dir = os.path.join(parent_dir, 'Data', 'RedditData', 'FilteredDatasets',theme,country)
       
      
        file = f"{theme}_data_{country}_{year}.csv"
        file_path = os.path.join(data_dir, file)
        model_names = self.model_names
    
        # Step 1: Translate content for each file sequentially
        translated_file_path = self.translate_content(file_path)
       
        # Step 2: Analyze sentiment for each translated file sequentially
        analyzed_file = self.analyze_sentiment(translated_file_path, model_names)
       

        return analyzed_file




    def save_to_csv(self, dataframe, file_path):
        if os.path.exists(file_path):
            dataframe.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8')
        else:
            dataframe.to_csv(file_path, mode='w', index=False, encoding='utf-8')




#------------------------------------------------Graph Functions------------------------------------------------------------
    def wordcloud(self, start_year,end_year,start_month,end_month, country,theme):
        nltk.download('stopwords')
        
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

        # Charger les données pour l'année et le pays spécifiés
        reddit_data = self.open_file_country(start_year,end_year,start_month,end_month, country,theme, False)
        
     

        # Déterminer la langue des mots vides
        # Ici on suppose que la langue des tweets est l'anglais
        stop_words_language = set(stopwords.words(lang_dict['en']))
        stop_words_en = set(stopwords.words('english'))

        # Fusionner tous les ensembles de mots vides pour obtenir un ensemble unique
        stop_words = stop_words_personal_set.union(stop_words_language).union(stop_words_en)

        # Dictionnaire pour stocker la fréquence des mots
        word_freq = {}

        # Boucle sur chaque tweet pour calculer la fréquence des mots
        for index, row in reddit_data.iterrows():
            try:
                post = row['content_translated']
            except:
                post = row['content']
            words = post.split()
            for word in words:
                word = clean_word(word)
                if word not in stop_words and word != '':
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Créer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Créer la figure et l'axe pour afficher le nuage de mots
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"Word Cloud from {start_month,start_year} to {end_month,end_year} in {country} with {len(reddit_data)} posts")
        ax.axis('off')

        # Encode the plot as a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(img.getvalue()).decode()
        
        return graph_base64,len(reddit_data)    


    def CircularGraph(self, start_year,end_year,start_month,end_month, country,theme):
        model_names = self.model_names
        
        reddit_data_analyzed = self.open_file_country(start_year,end_year,start_month,end_month, country,theme,True)
            
       
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#DEB7BC', '#75343C', '#A3A3A3']
        explode = (0.1, 0, 0)  # Explode the 1st slice

        for j, model_name in enumerate(model_names):
            model_id = model_name.split("/")[-1]
            try:
                positive_count_ai = (reddit_data_analyzed[f'{model_id}_sentiment'] == 2).sum()
                negative_count_ai = (reddit_data_analyzed[f'{model_id}_sentiment'] == 0).sum()
                neutral_count_ai = (reddit_data_analyzed[f'{model_id}_sentiment'] == 1).sum()
            except KeyError as e:
                print(f"KeyError: {e} not found in the data.")
                continue
            
            sizes = [positive_count_ai, negative_count_ai, neutral_count_ai]

            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            ax.set_title(f'Sentiment Distribution for Model: {model_id} ({start_year}-{end_year})')

            # Encode the plot as a base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

            graph_base64 = base64.b64encode(img.getvalue()).decode()
            
        return graph_base64,len(reddit_data_analyzed)

    
    def SentimentEvolution(self, rangeYear, country,theme):
        # Exemple de chargement des données de tweets pour les années données
        model_names = self.model_names
        model_name = model_names[0]  # Sélectionner le premier modèle pour l'exemple
        model_name_segment = model_name.split("/")[-1]
        sentiment_column = f"{model_name_segment}_sentiment"

        all_data = []

        start_year, end_year = rangeYear

        
        reddit_data_combined = self.open_file_country(start_year,end_year,1,12, country, theme,True)
        print(f"Sentiment data for {start_year}-{end_year} already exists.")
          
        # Calculer la moyenne des scores de sentiment par année
        sentiment_by_year = reddit_data_combined.groupby('year')[sentiment_column].mean().reset_index()

        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sentiment_by_year['year'], sentiment_by_year[sentiment_column], marker='o', linestyle='-')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Sentiment Score')
        ax.set_title(f'Average Sentiment Scores for {country.upper()} posts ({rangeYear})')
        ax.grid(True)
        plt.tight_layout()

        # Encoder le graphique comme une chaîne base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(img.getvalue()).decode()

        return graph_base64,len(reddit_data_combined)


   
    def plot_sentiment_map(self, start_year,end_year,start_month,end_month,theme):
        print('Plotting sentiment map...')
        current_dir = os.getcwd()

        # Remonte de deux niveaux pour arriver à 'FoodSafetyEngagement'
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

        # Ajoute les chemins nécessaires pour atteindre 'Data'
        data_dir = os.path.join(parent_dir, 'Data', 'RedditData', 'AnalyzedDatasets',theme)
     
        countries = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print('countriesare',countries)
        print(f"Found {len(countries)} countrie/s for the years {start_year}-{end_year}.")
        if not countries:
            print(f"No countries found for the year {start_year}-{end_year}.")
            return ""

        

        country_to_regions = {
            'france': ['France'], 'germany': ['Germany'], 'usa': ['United States of America'],
            'spain': ['Spain'], 'mexico': ['Mexico'], 'italy': ['Italy'], 'brazil': ['Brazil'],
            'portugal': ['Portugal'], 'netherlands': ['Netherlands'], 'russia': ['Russia'],
            'japan': ['Japan'], 'china': ['China'], 'india': ['India'], 'south_korea': ['South Korea'],
            'turkey': ['Turkey'], 'poland': ['Poland'], 'sweden': ['Sweden'], 'denmark': ['Denmark'],
            'norway': ['Norway'], 'finland': ['Finland'], 'unitedkingdom': ['United Kingdom'], 
        }

        all_data = []

       

        # Load and process the data
        for country in countries:
            try:
                reddit_data = self.open_file_country(start_year, end_year,start_month,end_month,country,theme, True)
                reddit_data['country'] = country
                regions = country_to_regions.get(country.lower(), [country])
                for index, row in reddit_data.iterrows():
                    for region in regions:
                        new_row = row.copy()
                        new_row['region'] = region
                        all_data.append(new_row)
            except Exception as e:
                print(f"Error processing data for {country}: {e}")

        tweets_data_extended = pd.DataFrame(all_data)

        model_name_segment = self.model_names[0].split("/")[-1]
        sentiment_column = f"{model_name_segment}_sentiment"

        region_sentiment = tweets_data_extended.groupby('region')[sentiment_column].mean().reset_index()

        world_shp_path = os.path.join(current_dir, 'ne_110m_admin_0_countries.shp')
        world = gpd.read_file(world_shp_path)

        world[sentiment_column] = -1

        for index, row in region_sentiment.iterrows():
            world.loc[world['NAME'] == row['region'], sentiment_column] = row[sentiment_column]

        def map_sentiment_label(score):
            if score == -1:
                return 'No Data'
            elif 0 <= score < 1:
                return 'Negative'
            elif 1 <= score < 1.5:
                return 'Neutral'
            elif score >= 1.5:
                return 'Positive'

        world['sentiment_label'] = world[sentiment_column].apply(map_sentiment_label)

        print("Number of regions with 'Neutral' sentiment:", (world['sentiment_label'] == 'Neutral').sum())
        print("Number of regions with 'Negative' sentiment:", (world['sentiment_label'] == 'Negative').sum())
        print("Number of regions with 'Positive' sentiment:", (world['sentiment_label'] == 'Positive').sum())
        print("Number of regions with no data:", (world['sentiment_label'] == 'No Data').sum())

        if world[sentiment_column].empty:
            raise ValueError("The sentiment_column is empty after joining. Check your data.")

        sentiment_cmap = plt.get_cmap('coolwarm')
        norm = Normalize(vmin=0, vmax=2)

        world['color'] = world[sentiment_column].apply(lambda x: sentiment_cmap(norm(x)) if x != -1 else 'white')

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world.plot(color=world['color'], ax=ax, linewidth=0.8, edgecolor='0.8')

        sm = ScalarMappable(cmap=sentiment_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])

        no_data_patch = mpatches.Patch(color='white', label='No Data')
        ax.legend(handles=[no_data_patch], loc='lower left')

        ax.set_title(f"World Map with Average Sentiment Scores ({start_year}-{end_year})")

        # Encode the plot as a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')  
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode(),len(tweets_data_extended)


    def get_every_comment(self, id, year, country,theme):
        reddit_data = self.open_file_country(year, country, theme,True)
        
        
        print(f"Loaded {len(reddit_data)} posts for {country}.")
        
        # Filtrer les lignes où parent_id est égal à l'id donné et où le type est comment
        comments = reddit_data[(reddit_data['parent_id'] == id) & (reddit_data['type'] == 'comment')]
        
        return comments


    def PostEvolution(self, rangeYear, country, theme):
        all_data = []

        start_year, end_year = rangeYear
        
        # Utiliser range avec ces variables
        for year in range(start_year, end_year + 1):
            try:
                reddit_data = self.open_file_country(rangeYear[0],rangeYear[1],1,12,country,theme, True)
                print(f"Loaded {len(reddit_data)} posts for {year} in {country}.")
            except Exception as e:
                print(f"Error opening file for {year}: {e}")
                continue  # Passer à l'année suivante si une erreur se produit

            reddit_data['year'] = year
            all_data.append(reddit_data)

        # Combiner tous les DataFrames en un seul
        reddit_data_combined = pd.concat(all_data, ignore_index=True)

        # Compter le nombre de posts par année
        posts_by_year = reddit_data_combined.groupby('year').size().reset_index(name='post_count')

        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(posts_by_year['year'], posts_by_year['post_count'], marker='o', linestyle='-', color='skyblue')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Posts')
        ax.set_title(f'Number of Posts for {country.upper()} in {theme} ({rangeYear})')
        ax.grid(axis='y')

        # Afficher uniquement les années sur l'axe des x
        ax.set_xticks(posts_by_year['year'])
        ax.set_xticklabels(posts_by_year['year'], rotation=45)  # Rotation pour une meilleure lisibilité

        plt.tight_layout()

        # Encoder le graphique comme une chaîne base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        graph_base64 = base64.b64encode(img.getvalue()).decode()

        return graph_base64,len(reddit_data_combined)
    
    
    def CountWordSentiment(self, start_year, end_year,start_month,end_month,country,theme):
        model_names = self.model_names
       
        reddit_data_analyzed = self.open_file_country(start_year, end_year,start_month,end_month,country,theme, True)

        stop_words = set(stopwords.words('english'))
        labels = ['Positive', 'Negative', 'Neutral']
        colors = ['#DEB7BC', '#75343C', '#A3A3A3']

        for model_name in model_names:
            model_id = model_name.split("/")[-1]
            try:
                content = reddit_data_analyzed['content_translated']
                sentiments = reddit_data_analyzed[f'{model_id}_sentiment']
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
                word_data = reddit_data_analyzed[content.str.contains(r'\b' + re.escape(word) + r'\b', regex=True)]
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
            ax.set_title(f'Top 10 Words by Sentiment for Model: {model_id} ({start_year}-{end_year})')
            ax.legend()

            # Encode the plot as a base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

            graph_base64 = base64.b64encode(img.getvalue()).decode()
            
        return graph_base64,len(reddit_data_analyzed)



        
if __name__ == "__main__":
    analyzer = RedditSentimentAnalyzerBack()
    # china_reddit_data = analyzer.open_file_country(2021, 'china', False)
    # non_empty_body_count = (china_reddit_data['body'].notnull() & (china_reddit_data['body'] != '')).sum()

    # print(non_empty_body_count)
    # print(china_reddit_data['type'].value_counts())