#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

"""
Main script to extract the features from all Gutemberg books and store it in a csv file

"""
import os
from textblob import TextBlob
import textstat
import re
import csv
from pandas.core.common import flatten
import numpy as np

import html_parser

# Global parameters
BOOKS_PATH = "books"
CHAPTER_SIZE = 3000
OUTPUT_FILE_NAME = "features.csv"
CSV_HEADERS = ["BOOK_ID", "Senti_S1", "Senti_S2", "Senti_S3", "Senti_E1", "Senti_E2", "Senti_E3", "S_count", "Avg_S_len", "Flesch" , "W_count", "Noun_Cnt"]
SENTIMENT_WEIGHT = 100


def start_end_sentiment(blob):
    chapter_parts_count = 3
    seperator = ' '
    chapter_part_size = int(CHAPTER_SIZE / chapter_parts_count)

    sentiment_start = []
    sentiment_end = []

    for i in range(0, CHAPTER_SIZE, chapter_part_size):
        chapter_part = TextBlob(seperator.join(blob.words[i:i + chapter_part_size]))
        sentiment_start.append(chapter_part.sentiment.polarity)

    for i in range(1, CHAPTER_SIZE + 1, chapter_part_size):
        chapter_part = TextBlob(seperator.join(blob.words[-i - chapter_part_size:-i]))
        sentiment_end.append(chapter_part.sentiment.polarity)

    if SENTIMENT_WEIGHT != 0:
        sentiment_start = [(x+1)*100 for x in sentiment_start]
        sentiment_end = [(x + 1) * 100 for x in sentiment_end]

    return sentiment_start, sentiment_end


def book_structure(blob, path_to_file):

    sentences = blob.sentences
    sentences_count = len(sentences)
    word_count = len(blob.words)
    proper_noun_count = len(np.unique([x[0] for x in blob.tags if x[1] == 'NNP']))
    # print(proper_noun_count)
    # nouns_count = len(blob.noun_phrases)
    # paragraph_count = html_parser.get_paragraph_count(path_to_file)

    avg_sentence_len = []
    for sentence in sentences:
        avg_sentence_len.append(len(sentence.words))

    s_len = len(avg_sentence_len)
    if s_len > 0:
        avg_sentence_len = sum(avg_sentence_len) / s_len
    else:
        avg_sentence_len = 0

    return sentences_count, avg_sentence_len , word_count, proper_noun_count #, nouns_count


def write_to_csv(data, mode):
    with open(OUTPUT_FILE_NAME, mode=mode, newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(data)


def extractor(path_to_file):
    book_data = html_parser.extract_text(path_to_file)
    blob = TextBlob(book_data)

    sentiment_start, sentiment_end = start_end_sentiment(blob)
    sentences_count, avg_sentence_len, word_count, proper_noun_count = book_structure(blob, path_to_file)
    flesch_score = textstat.flesch_reading_ease(book_data)

    return sentiment_start, sentiment_end, sentences_count, avg_sentence_len, flesch_score ,word_count, proper_noun_count


if __name__ == "__main__":

    print("Initializing feature extractor ...")
    # checking if the output csv already exists
    if os.path.exists(OUTPUT_FILE_NAME):
        print("ERROR: A file with the name ", OUTPUT_FILE_NAME, " already exists ! move the file to another directory or rename the file and try again")
        exit(-1)

    # Gets a list of all books
    books_list = os.listdir(BOOKS_PATH)
    books_count = len(books_list)
    # print(books_list)
    # print(html_parser.extract_text(os.path.join(BOOKS_PATH, books_list[0])))

    write_to_csv(CSV_HEADERS, "w")

    # books_list = ["pg34164-content.html"]
    idx = 1
    for book in books_list:
        features = []
        book_id = re.findall('(^pg[0-9]*)', book)
        features.append(book_id)
        print(idx, "/", books_count, " - extracting from ", book_id)
        path_to_file = os.path.join(BOOKS_PATH, book)
        for feature in extractor(path_to_file):
            features.append(feature)
        idx += 1
        write_to_csv(list(flatten(features)), "a")
