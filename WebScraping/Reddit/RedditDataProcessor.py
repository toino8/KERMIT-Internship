import os
import json
import csv
import logging
import logging.handlers
from datetime import datetime, timezone
import zstandard


""" Here is the class to treat the subreddit downloaded on academic torrents as zst files in zipped folders.
We are basically filtering data per year (and of course per country) and saving the data in csv files.
The reason we have to split documents for different years is that the data is too big to be treated in one go.
This code is the first step of the data treatment process when you download torrents from academic torrents.
The result is saved in a new csv file in the 'RawDatasets' folder."""



class RedditDataProcessor:
    def __init__(self, data_torrent_dir=None, data_raw_dir=None, countries=None, output_format="csv", single_field=None, write_bad_lines=True, field=None, values=[''], values_file=None, exact_match=False):
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir, os.pardir))
        self.data_torrent_dir = data_torrent_dir if data_torrent_dir else os.path.join(self.parent_dir, "Data", "RedditData", "Torrented")
        self.data_raw_dir = data_raw_dir if data_raw_dir else os.path.join(self.parent_dir, "Data", "RedditData", "RawDatasets")
        self.countries = countries if countries else [folder for folder in os.listdir(self.data_torrent_dir) if os.path.isdir(os.path.join(self.data_torrent_dir, folder))]
        self.output_format = output_format
        self.single_field = single_field
        self.write_bad_lines = write_bad_lines
        self.field = field
        self.values = values
        self.values_file = values_file
        self.exact_match = exact_match
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

    def write_line_zst(self, handle, line):
        handle.write(line.encode('utf-8'))
        handle.write("\n".encode('utf-8'))

    def write_line_json(self, handle, obj):
        handle.write(json.dumps(obj))
        handle.write("\n")

    def write_line_single(self, handle, obj, field):
        if field in obj:
            handle.write(obj[field])
        else:
            self.log.info(f"{field} not in object {obj['id']}")
        handle.write("\n")

    def write_line_csv(self, writer, obj, is_submission):
        output_list = []
        output_list.append(obj.get('id', ''))
        output_list.append(obj.get('parent_id', ''))
        output_list.append(obj.get('link_id', ''))
        output_list.append(obj.get('author', ''))
        output_list.append(str(obj.get('score', '')))
        output_list.append(datetime.fromtimestamp(int(obj.get('created_utc', 0))).strftime("%Y-%m-%d"))
        
        if is_submission:
            title = obj.get('title', '')
            selftext = obj.get('selftext', '')
            url = obj.get('url', '')
            
            output_list.append(title)
            output_list.append(selftext)
            output_list.append('')  # Body is not applicable for submissions
            
            # Combine title with selftext or url
            content = f"{title}\n\n{selftext}" if obj.get('is_self', False) else f"{title}\n\n{url}"
        else:
            output_list.append('')  # Title is not applicable for comments
            output_list.append('')  # Selftext is not applicable for comments
            output_list.append(obj.get('body', ''))
            content = obj.get('body', '')
        
        output_list.append(f"https://www.reddit.com{obj.get('permalink', '')}")
        output_list.append(content)
        
        writer.writerow(output_list)

    def read_and_decode(self, reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
        chunk = reader.read(chunk_size)
        bytes_read += chunk_size
        if previous_chunk is not None:
            chunk = previous_chunk + chunk
        try:
            return chunk.decode()
        except UnicodeDecodeError:
            if bytes_read > max_window_size:
                raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
            self.log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
            return self.read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

    def read_lines_zst(self, file_name):
        with open(file_name, 'rb') as file_handle:
            buffer = ''
            reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(file_handle)
            while True:
                chunk = self.read_and_decode(reader, 2 ** 27, (2 ** 29) * 2)

                if not chunk:
                    break
                lines = (buffer + chunk).split("\n")

                for line in lines[:-1]:
                    yield line.strip(), file_handle.tell()

                buffer = lines[-1]

            reader.close()

    def process_file(self, input_file, output_file_base, output_format, field, values, single_field, exact_match):
        is_submission = "submission" in input_file
        self.log.info(f"Input: {input_file} : Is submission {is_submission}")
        writer = None
        
        file_size = os.stat(input_file).st_size
        created = None
        matched_lines = 0
        bad_lines = 0
        total_lines = 0
        
        for line, file_bytes_processed in self.read_lines_zst(input_file):
            total_lines += 1
            if total_lines % 100000 == 0:
                self.log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : {matched_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

            try:
                obj = json.loads(line)
                created = datetime.fromtimestamp(int(obj['created_utc']), tz=timezone.utc)
                year = created.year
                
                output_file = f"{output_file_base}_{year}.{output_format}"
                # Ensure the directory for the output file exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                if not writer or writer.output_file != output_file:
                    if writer:
                        writer.handle.close()
                    writer = open(output_file, 'a', encoding='UTF-8', newline='')
                    csv_writer = csv.writer(writer)
                    # Écrire l'en-tête général une seule fois
                    if writer.tell() == 0:  # Vérifie si le fichier est vide
                        csv_writer.writerow(['id', 'parent_id', 'link_id',  'author', 'score', 'created', 'title', 'selftext', 'body', 'permalink', 'content'])
                    writer.csv_writer = csv_writer
                    writer.output_file = output_file
                    writer.handle = writer

                if field is not None:
                    field_value = obj[field].lower()
                    matched = False
                    for value in values:
                        if exact_match:
                            if value == field_value:
                                matched = True
                                break
                        else:
                            if value in field_value:
                                matched = True
                                break
                    if not matched:
                        continue

                matched_lines += 1
                self.write_line_csv(writer.csv_writer, obj, is_submission)
                
            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1
                if self.write_bad_lines:
                    if isinstance(err, KeyError):
                        self.log.warning(f"Key {field} is not in the object: {err}")
                    elif isinstance(err, json.JSONDecodeError):
                        self.log.warning(f"Line decoding failed: {err}")
                    self.log.warning(line)

        if writer:
            writer.handle.close()
        self.log.info(f"Complete : {total_lines:,} : {matched_lines:,} : {bad_lines:,}")

    def process_data(self):
        self.log.info("Starting data processing for academic torrents data")
        if self.single_field is not None:
            self.log.info("Mode de sortie sur un seul champ, changement du format de fichier de sortie en txt")
            self.output_format = "txt"

        if self.values_file is not None:
            self.values = []
            with open(self.values_file, 'r') as values_handle:
                for value in values_handle:
                    self.values.append(value.strip().lower())
            self.log.info(f"Chargé {len(self.values)} valeurs à partir du fichier de valeurs {self.values_file}")
        else:
            self.values = [value.lower() for value in self.values]  # Convertir en minuscules

        self.log.info(f"Filtrage sur le champ : {self.field}")
        if len(self.values) <= 20:
            self.log.info(f"Sur les valeurs : {','.join(self.values)}")
        else:
            self.log.info("Sur les valeurs :")
            for value in self.values:
                self.log.info(value)

        for country in self.countries:
            if country in os.listdir(self.data_raw_dir):
                self.log.info(f"The country directory {country} already exists. Skipping to the next country.")
                print(f'The country directory {country} already exists. Skipping to the next country.')
                continue

            output_dir = os.path.join(self.data_raw_dir, country)
            input_dir = os.path.join(self.data_torrent_dir, country)
            output_file_country = os.path.join(output_dir, f"reddit_data_{country}")
            input_files = []

            if os.path.isdir(input_dir):
                self.log.info(f"Processing directory: {input_dir}")
                for file in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, file)
                    if os.path.isfile(file_path) and file.endswith(".zst"):
                        input_name = os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]
                        input_files.append((file_path, output_file_country))
            else:
                input_files.append((input_dir, output_file_country))

            self.log.info(f"Processing {len(input_files)} files for {country}")
            for file_in, file_out in input_files:
                self.process_file(file_in, file_out, self.output_format, self.field, self.values, self.single_field, self.exact_match)

if __name__ == "__main__":
    processor = RedditDataProcessor()
    processor.process_data()

