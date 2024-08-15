from flask import Flask, request, jsonify
from flask_cors import CORS
from RedditSentimentAnalyzer import RedditSentimentAnalyzerBack
from TweetSentimentAnalyzer import TweetSentimentAnalyzer
import os

app = Flask(__name__)
CORS(app, resources={r"/get-countries": {"origins": "http://localhost:3000"}, 
                      r"/run-script": {"origins": "http://localhost:3000"}})

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    print('Received payload:', data)  # Log the received payload
    country = data.get('country')
    yearMonths = data.get('yearMonths')  # List of month-year objects
    month_mapping = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }
 

    if yearMonths:
        start_year = min(item['year'] for item in yearMonths)
        end_year = max(item['year'] for item in yearMonths)
        # Calcul de start_month et end_month
        start_month = min(month_mapping[item['month']] for item in yearMonths if item['year'] == start_year)
        end_month = max(month_mapping[item['month']] for item in yearMonths if item['year'] == end_year)

   

    # Remaining data processing...
    rangeYearEvolution = data.get('rangeYear')
    countryEvolution = data.get('countryEvolution')
    rangeYearPostEvolution = data.get('rangeYearPostEvolution')
    postEvolutionCountry = data.get('postEvolutionCountry')
    postEvolution = data.get('postEvolution')
    
    reddit_sentiment_map = data.get('redditSentimentMap')
    reddit_circular_graph = data.get('redditCircularGraph')
    reddit_sentiment_evolution = data.get('redditSentimentEvolution')
    redditWordCloud = data.get('redditWordCloud')
    redditWordCloudCountry = data.get('redditWordCloudCountry') 
    redditGraphCountry = data.get('redditGraphCountry')
    redditWordCount = data.get('redditWordCount')
    redditWordCountCountry = data.get('redditWordCountCountry')
    
    twitter_sentiment_map = data.get('twitterSentimentMap')
    twitter_circular_graph = data.get('twitterCircularGraph')
    twitter_sentiment_evolution = data.get('twitterSentimentEvolution')
    twitterWordCloud = data.get('twitterWordCloud')
    twitterWordCloudLanguage = data.get('twitterWordCloudLanguage')
    twitterGraphLanguage = data.get('twitterGraphLanguage')
    twitterWordCount = data.get('twitterWordCount')
    twitterWordCountLanguage = data.get('twitterWordCountLanguage')
    
    theme_chosen = data.get('theme_chosen')

    current_dir = os.getcwd()

    # Remonte de deux niveaux pour arriver à 'FoodSafetyEngagement'
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    # Ajoute les chemins nécessaires pour atteindre 'Data'
    analyzed_dir = os.path.join(parent_dir, 'Data', 'RedditData', 'AnalyzedDatasets', theme_chosen)
    countries_analyzed = [d for d in os.listdir(analyzed_dir) if os.path.isdir(os.path.join(analyzed_dir, d))]
    data_dir = os.path.join(parent_dir, 'Data', 'RedditData', 'FilteredDatasets', theme_chosen)
    countries_data = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
   
    try:
        graphs = {}  # Use a dictionary to store multiple graphs
        number_posts = {}
        
        # Replace year ranges with extracted start and end year/month
        if twitter_sentiment_evolution:
            analyzer = TweetSentimentAnalyzer()
            print(f'Running Twitter sentiment evolution for range {rangeYearEvolution} and country {countryEvolution}')
            key = f'twitterSentimentEvolution_{rangeYearEvolution}_{countryEvolution}'
            output = analyzer.SentimentEvolution(rangeYearEvolution, countryEvolution)
            graphs[key] = output[0]
            number_posts[key] = output[1]
                
        if reddit_sentiment_evolution:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit sentiment evolution for range {rangeYearEvolution} and country {countryEvolution}')
            key = f'redditSentimentEvolution_{rangeYearEvolution}_{countryEvolution}'
            output = analyzer.SentimentEvolution(rangeYearEvolution, countryEvolution,theme_chosen)
            graphs[key] = output[0]
            number_posts[key] = output[1]
                
        if postEvolution:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit post evolution for range {rangeYearPostEvolution} and country {postEvolutionCountry}')
            key = f'postEvolution_{rangeYearPostEvolution}_{postEvolutionCountry}'
            output = analyzer.PostEvolution(rangeYearPostEvolution, postEvolutionCountry,theme_chosen)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        
        if redditWordCount:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit word count for years {start_year}-{end_year} and country {redditWordCountCountry}')
            output = analyzer.CountWordSentiment(start_year,end_year,start_month,end_month, redditWordCountCountry,theme_chosen)
            key = f'redditWordcount_{start_year}-{end_year}_{redditWordCountCountry}'
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if twitterWordCount:
            analyzer = TweetSentimentAnalyzer()
            print(f'Running Twitter word count for years {start_year}-{end_year} and language {twitterWordCountLanguage}')
            output = analyzer.CountWordSentiment(start_year,end_year,start_month,end_month, twitterWordCountLanguage)
            key = f'twitterWordcount_{start_year}-{end_year}_{twitterWordCountLanguage}'
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if redditWordCloud:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit word cloud for years {start_year}-{end_year} and country {redditWordCloudCountry}')
            key = f'redditWordcloud_{start_year}-{end_year}_{redditWordCloudCountry}'
            output = analyzer.wordcloud(start_year,end_year,start_month,end_month, redditWordCloudCountry,theme_chosen)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if twitterWordCloud:
            analyzer = TweetSentimentAnalyzer()
            print(f'Running Twitter word cloud for years {start_year}-{end_year} and language {twitterWordCloudLanguage}')
            key = f'twitterWordcloud_{start_year}-{end_year}_{twitterWordCloudLanguage}'
            output = analyzer.wordcloud(start_year,end_year,start_month,end_month, twitterWordCloudLanguage)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        
            
        if reddit_sentiment_map:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit sentiment map for yearw {start_year}-{end_year}')
            key = f'redditSentimentMap_{start_year}-{end_year}'
            output = analyzer.plot_sentiment_map(start_year,end_year,start_month,end_month,theme_chosen)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if reddit_circular_graph:
            analyzer = RedditSentimentAnalyzerBack()
            print(f'Running Reddit circular graph for years {start_year}-{end_year} and country {redditGraphCountry}')
            key = f'redditCircularGraph_{start_year}-{end_year}_{redditGraphCountry}'
            output = analyzer.CircularGraph(start_year,end_year,start_month,end_month, redditGraphCountry,theme_chosen)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if twitter_sentiment_map:
            analyzer = TweetSentimentAnalyzer()
            print(f'Running Twitter sentiment map for years {start_year}-{end_year} and country {country}')
            key = f'twitterSentimentMap_{start_year}-{end_year}_{country}'
            output = analyzer.plot_sentiment_map(start_year,end_year,start_month,end_month, country)
            graphs[key] = output[0]
            number_posts[key] = output[1]
            
        if twitter_circular_graph:
            analyzer = TweetSentimentAnalyzer()
            print(f'Running Twitter circular graph for years {start_year}-{end_year} and language {twitterGraphLanguage}')
            key = f'twitterCircularGraph_{start_year}-{end_year}_{twitterGraphLanguage}'
            output = analyzer.CircularGraph(start_year,end_year,start_month,end_month, twitterGraphLanguage)
            graphs[key] = output[0]
            number_posts[key] = output[1]

        return jsonify({'graphs': graphs, 'number_posts': number_posts, 'countries_analyzed' :countries_analyzed,'countries_data':countries_data,'error': ''})
    except Exception as e:
        print(f'Error occurred: {e}')  # Log error for debugging
        return jsonify({'graphs': {}, 'error': str(e)}), 400




if __name__ == '__main__':
    app.run(debug=True)
