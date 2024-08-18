# FootSafetyEngagement


**Public Engagement in Environmental Issues: First focus on Climate Change and Microplastics**

*Leveraging Artificial Intelligence for Big Data Analysis to Gauge Consumer Sentiment*

## Overview

The PublicEngagementEnvironment project focuses on analyzing public engagement and consumer sentiment related to environmental themes such as climate change and microplastics.
This is accomplished by collecting and analyzing large datasets from various online platforms, utilizing Natural Language Processing (NLP) and machine learning techniques to extract meaningful insights.

### Project Workflow

1. **Data Collection and Exploratory Data Analysis**:  
   We gather data from a wide array of web sources, including social media platforms, reviews, blogs, and online forums. This comprehensive data collection forms the foundation for our analysis.

2. **Natural Language Processing (NLP) and Machine Learning Model Development**:  
   Utilizing advanced NLP tools and techniques, we preprocess and analyze the textual data collected. We then apply classification algorithms to categorize sentiments and extract relevant information from the data.

3. **Evaluation and Contextualization of Results**:  
   The findings are visualized to present insights into consumer attitudes, dominant narratives, and the interplay between climate change and food safety concerns. This step helps contextualize the data in a way that is easy to understand and actionable.

## Setup Instructions

### Prerequisites

Only if you want to modify the Twitter Webscraping :
Before running any code, ensure that you have the Chrome WebDriver installed. You can download it from [here](https://storage.googleapis.com/chrome-for-testing-public/125.0.6422.76/win64/chromedriver-win64.zip). Additionally, make sure that Google Chrome is installed on your machine.

For Reddit, while I am not sure that I can provide you the analyzed data to avoid you lacking of time , or if you want to personalize the analysis by specifying different keywords on a theme or a different theme, go on https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10 and download it with a torrent client, then don't forget to specify the subreddit of the country you want , for example type France and select france_sumbissions and france_comments, then you take the two zst files and put it in a folder with the name of the country in Data/RedditData/Torrented/{country}

### Environment Variables

Only if you want to use the Twitter Webscraping :  
To keep your Twitter account credentials secure, create a `.env` file in the root directory of the project and include your **EMAIL_ADDRESS**, **PASSWORD**, and **PHONE_NUMBER** associated with your Twitter account. Since this file is included in the `.gitignore`, it will not be pushed to GitHub, ensuring that your credentials remain private.

## Project Structure

The project consists of two main components: Reddit and Twitter data processing.

### Data Collection

#### Reddit Data

Be sure to have the prerequisites for reddit data ( torrents files ) before running the notebook associated with theses files 

The Reddit data is obtained from academic torrents and is stored in the project repository. Inside the **WebScrapping/Reddit** folder, you will find the following Python scripts:

- **RedditDataProcessor.py**: This script is responsible for extracting data from ZIP and ZST files downloaded from academic torrents.
  
- **RedditScraper2204.py**: Designed specifically for the year 2024, this script handles the extraction of relevant Reddit data, as academic torrents currently provide access only up to 2023.
  
- **RedditThemeFilter.py**: This script allows you to filter the data into specified themes based on keywords you provide.


To execute these scripts, you can utilize the Jupyter Notebook named **DataTreatment.ipynb**. This notebook orchestrates the entire data processing workflow, running all the necessary scripts sequentially.

#### Twitter Data

Currently, the Twitter data collection framework is not implemented, but it will be integrated in future updates.

### Data Analysis

After collecting the data, we proceed to translate and analyze the content using a machine learning model. You will find the **LaunchTranslationSentimentAnalysis.ipynb** notebook in the **UI** folder. This notebook is responsible for analyzing and translating all collected data files. Note that due to rate limits imposed by Hugging Face, you may need to rerun this notebook if you encounter any issues.

### User Interface

To launch the user interface, open two terminal windows:

1. In the **backend** folder, run:
   ```bash
   python app.py
    ```

2. In the **UI** folder, run:
   ```bash
   npm run dev
    ```

This will launch the user interface, allowing navigation through:

- **Home Page**: Access functionalities to generate graphics with the data.
- **About Page**: Information about the project and its components with the graphics.
- **Results Page**: Visualizations of the analyzed data.

The UI is built using React (JavaScript) for the frontend and Flask (Python) for the backend.

### Contact
For any questions or support, please contact me at theo.hint@gmail.com.

Thank you for your interest in the Food Safety Engagement project!
