import os
import nltk
nltk.download('averaged_perceptron_tagger')
import string
from gensim.models import Word2Vec
import math
import random
from collections import defaultdict

# add all song files in path to list
data_folder = os.path.join(os.getcwd(), 'Homework 3 Data')
country_folder = os.path.join(data_folder, 'Country')
rock_folder = os.path.join(data_folder, 'Rock')
pop_folder = os.path.join(data_folder, 'Pop')
metal_folder = os.path.join(data_folder, 'Metal')
genre_folders = [country_folder, rock_folder, pop_folder]


# method to make lower case, remove punctuation, tokenize
def clean(data):
    data = data.lower()
    data = "".join(char for char in data if char not in string.punctuation)
    data = nltk.word_tokenize(data)  # data is a list now
    return data


# print each individual song from list
def print_songlist(genre_folder):
    songlist = os.listdir(genre_folder)
    for filename in songlist:
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(genre_folder, filename), "r") as file_handler:
            data = file_handler.read()
            print("\n" + filename)
            data = clean(data)
            print("\n", data)

country_songs = []
pop_songs = []
rock_songs = []
country_labeled_songs = []
pop_labeled_songs = []
rock_labeled_songs = []

# label lyrics as belonging to a certain genre
def label_song_lyrics(genre_folder):
    songlist = os.listdir(genre_folder)
    for filename in songlist:
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(genre_folder, filename), "r") as file_handler:
            data = file_handler.read()
            lyrics = clean(data)
            if genre_folder == country_folder:
                country_labeled_songs.append(('country', lyrics))
                country_songs.append(lyrics)
            elif genre_folder == pop_folder:
                pop_labeled_songs.append(('pop', lyrics))
                pop_songs.append(lyrics)
            elif genre_folder == rock_folder:
                rock_labeled_songs.append(('rock', lyrics))
                rock_songs.append(lyrics)
for genre in genre_folders:
    label_song_lyrics(genre)

# generate list of all lyrics per genre
def lyrics_only(genre_songs):
    lyrics = []
    for song in genre_songs:
        for word in song[1]:
            if word not in lyrics:
                lyrics.append(word)
    return lyrics

country_lyrics = lyrics_only(country_labeled_songs)
pop_lyrics = lyrics_only(pop_labeled_songs)
rock_lyrics = lyrics_only(rock_labeled_songs)

# determine how each genre uniquely uses parts of speech
country_tags = nltk.pos_tag(country_lyrics)
pop_tags = nltk.pos_tag(pop_lyrics)
rock_tags = nltk.pos_tag(rock_lyrics)
country_tags_dict = defaultdict(list)
pop_tags_dict = defaultdict(list)
rock_tags_dict = defaultdict(list)
for word, tag in country_tags:
    country_tags_dict[tag].append(word)
for word, tag in pop_tags:
    pop_tags_dict[tag].append(word)
for word, tag in rock_tags:
    rock_tags_dict[tag].append(word)

# select a random song from each genre to serve as a template with regards to POS_tags
rand = random.randint(0, 99)
country_template_song = country_labeled_songs[rand][1]
rock_template_song = rock_labeled_songs[rand][1]
pop_template_song = pop_labeled_songs[rand][1]

# tag template songs for parts of speech
country_tagged_template = nltk.pos_tag(country_template_song)
rock_tagged_template = nltk.pos_tag(rock_template_song)
pop_tagged_template = nltk.pos_tag(pop_template_song)

# generate list of only parts of speech in same order as template song
country_template_tags = [tag for (word, tag) in country_tagged_template]
rock_template_tags = [tag for (word, tag) in rock_tagged_template]
pop_template_tags = [tag for (word, tag) in pop_tagged_template]


# initialize and train models specific to each genre
country_model = Word2Vec(country_songs, min_count=1)
country_model.build_vocab(country_songs)
country_model.train(country_songs, total_examples=country_model.corpus_count, epochs=country_model.epochs)

pop_model = Word2Vec(pop_songs, min_count=1)
pop_model.build_vocab(pop_songs)
pop_model.train(pop_songs, total_examples=country_model.corpus_count, epochs=country_model.epochs)

rock_model = Word2Vec(rock_songs, min_count=1)
rock_model.build_vocab(rock_songs)
rock_model.train(rock_songs, total_examples=country_model.corpus_count, epochs=country_model.epochs)


def generate_lyrics(model, tags_template, genre_tags_dict):
    new_song = []
    if len(new_song) == 0:
        first_tag = tags_template[0]
        options = genre_tags_dict[first_tag]
        rand_num = random.randint(0, len(options) - 1)
        rand_word = options[rand_num]
        new_song.append(model.wv.most_similar(rand_word)[0][0])
    for i in range(1, len(tags_template)-1):
        next_tag = tags_template[i]
        options = genre_tags_dict[next_tag]
        try :
            rand_num = random.randint(0, len(options) - 1)
            rand_word = options[rand_num]
        except : continue
        new_song.append(model.wv.most_similar(rand_word)[0][0])
    return new_song

generated_country_lyrics = generate_lyrics(country_model, country_template_tags, country_tags_dict)
generated_pop_lyrics = generate_lyrics(pop_model, pop_template_tags, pop_tags_dict)
generated_rock_lyrics = generate_lyrics(rock_model, rock_template_tags, rock_tags_dict)


print("Country Song: " + str(generated_country_lyrics))
print("Pop Song: " + str(generated_pop_lyrics))
print("Rock Song: " + str(generated_rock_lyrics))

