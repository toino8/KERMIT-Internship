import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
import logging.handlers

class RedditThemeFilter:
    def __init__(self, theme, words_theme):
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
            'italy': 'it',          # Italien
            'unitedkingdom': 'en',  # Anglais   
        }
        self.current_dir = os.getcwd()
        self.great_parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir, os.pardir))
        self.data_dir = os.path.join(self.great_parent_dir, "Data", "RedditData", "RawDatasets")
        self.theme = theme
        self.words_theme = words_theme
        self.log = self.setup_logger()

    def setup_logger(self):
        log = logging.getLogger("bot")
        log.setLevel(logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        log_str_handler = logging.StreamHandler()
        log_str_handler.setFormatter(log_formatter)
        log.addHandler(log_str_handler)
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024 * 1024 * 16, backupCount=5)
        log_file_handler.setFormatter(log_formatter)
        log.addHandler(log_file_handler)
        return log
    
    
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
            self.log.info(f"Translation failed: {e}")
            return None

    def process_country_data(self, country):
        country_dir = os.path.join(self.data_dir, country)
        save_dir = os.path.join(self.great_parent_dir, "Data", "RedditData", "FilteredDatasets", self.theme, country)
        
        # Skip processing if the save directory already exists
        if os.path.exists(save_dir):
            self.log.info(f"Data for {country} already exists. Skipping to the next country.")
            return
        
        # Create the save directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        data_files = [f for f in os.listdir(country_dir) if f.endswith('.csv')]
        self.log.info(f"Processing {len(data_files)} files for {country}...")

        for file in data_files:
            file_path = os.path.join(country_dir, file)
            
            # Extract the year from the filename
            year = file.split('_')[3].replace('.csv', '')

            data = pd.read_csv(file_path)
            self.log.info(f"Loaded {len(data)} posts from {file}.")

            language = self.country_language_map.get(country.lower(), 'en')

            # Translate the theme words if the language is not English
            if language != 'en':
                translated_words = [self.translate_sentence(word, language) for word in self.words_theme] + self.words_theme
            else:
                translated_words = self.words_theme
                
            data['title'] = data['title'].astype(str)
           
            
            # Create a mask to select rows containing the theme words
            mask = data['title'].str.contains('|'.join(translated_words), case=False, na=False) | \
                   data['content'].str.contains('|'.join(translated_words), case=False, na=False)
            
            # Filter the data
            theme_data = data[mask].copy()  # Create a copy to avoid the SettingWithCopyWarning
            
            # Skip to the next file if there's no relevant data
            if theme_data.empty:
                self.log.info(f"No relevant posts found in {file} with the theme {self.theme}.")
                continue
            else:
                self.log.info(f"Found {len(theme_data)} posts containing the theme {self.theme} in {file}.")
            
            # Save the filtered data to a CSV file, grouped by year
            current_year_file = os.path.join(save_dir, f"{self.theme}_data_{country}_{year}.csv")
            
            if os.path.exists(current_year_file):
                self.log.info(f"Data for {country} in {year} already exists.")
                continue
                
            theme_data.to_csv(current_year_file, index=False, encoding='utf-8')
            self.log.info(f"Saved {len(theme_data)} posts for {country} in {year} to {current_year_file}")

    def create_csvs_with_theme(self):
        country_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.log.info(f"Found {len(country_dirs)} country directories.")
        
        for country in country_dirs:
            self.process_country_data(country)

# Usage example
if __name__ == "__main__":
    theme = 'climatechange'
    words_theme = ['climate', 'climate change']
    filter_instance = RedditThemeFilter(theme, words_theme)
    filter_instance.create_csvs_with_theme()
