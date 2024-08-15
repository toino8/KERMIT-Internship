import praw
import pandas as pd
import os
import time
import logging.handlers

class RedditDataDownloader:
    def __init__(self, client_id="0pCL1CobpSuuFO4j0HtHFA", client_secret="KZGvz5LVVSR4ziyvDwxYfs8L2n0jtg", user_agent="university-project", countries=['unitedkingdom'], save_dir=None):
        self.reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent=user_agent)
        self.countries = countries
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir, os.pardir))
        self.save_dir = save_dir if save_dir else os.path.join(self.parent_dir, "Data", "RedditData", "RawDatasets")
        self.log = self.setup_logger()
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
    
    def wait_and_retry(self, retries, delay):
        """Wait and retry if rate limit error occurs."""
        for attempt in range(retries):
            try:
                yield
                return
            except praw.exceptions.APIException as e:
                if e.error_type == 'RATELIMIT':
                    self.log.info(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise
        raise Exception("Max retries exceeded")

    def download_subreddit_posts(self, country, max_limit=1000000):
        all_posts = []
        limit_per_request = 1000000  # Adjust as necessary

        try:
            subreddit = self.reddit.subreddit(country)
            for submission in self.wait_and_retry(retries=5, delay=1):
                for submission in subreddit.new(limit=None):
                    post_data = {
                        'id': submission.id,
                        'parent_id': None,
                        'title': submission.title,
                        'score': submission.score,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'created': submission.created_utc,
                        'selftext': submission.selftext,
                        'content': submission.selftext,  # Changed 'body' to 'content'
                        'type': 'submission'
                    }
                    all_posts.append(post_data)

                    submission.comments.replace_more(limit=None)
                    for comment in submission.comments.list():
                        comment_data = {
                            'id': comment.id,
                            'parent_id': comment.parent_id.split('_')[-1],
                            'title': None,
                            'score': comment.score,
                            'url': comment.permalink,
                            'num_comments': None,
                            'created': comment.created_utc,
                            'body': comment.body,
                            'content': comment.body,  # Changed 'body' to 'content'
                            'type': 'comment'
                        }
                        all_posts.append(comment_data)

                    if len(all_posts) >= max_limit:
                        return all_posts

                    time.sleep(0.5)  # Adjust sleep time to avoid rate limits

        except Exception as e:
            self.log.info(f"Failed to download subreddit posts for {country}: {e}")
        return all_posts

    def download_and_save_all(self):
        self.log.info("Downloading Reddit data for 2024 using praw")
        for country in self.countries:
            csv_filename = f"reddit_data_{country}_2024.csv"
            new_save_dir = os.path.join(self.save_dir, country) 
            csv_filepath = os.path.join(new_save_dir, csv_filename)
            if os.path.exists(csv_filepath):
                self.log.info(f"File already exists for {country}. Skipping...")
                continue
            self.log.info(f"Downloading posts for {country}...")

            posts = self.download_subreddit_posts(country)

            if posts:
                df = pd.DataFrame(posts)
                df['created'] = pd.to_datetime(df['created'], unit='s')  # Convert timestamp to datetime
                df.to_csv(csv_filepath, index=False, encoding='utf-8')
                self.log.info(f"Saved {len(posts)} posts for {country} to {csv_filepath}")
            else:
                self.log.info(f"No posts found for {country}")
