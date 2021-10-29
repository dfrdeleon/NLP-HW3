import os
import nltk
import string

# add all song files in path to list
countrysongs = "/Users/annakim/Documents/Homework3/Country"
rocksongs = "/Users/annakim/Documents/Homework3/Rock"
popsongs = "/Users/annakim/Documents/Homework3/Pop"
metalsongs = "/Users/annakim/Documents/Homework3/Metal"
songlist = os.listdir(countrysongs)


# method to make lower case, remove punctuation, tokenize
def clean(data) :
    data = data.lower()
    data = "".join(char for char in data if char not in string.punctuation)
    data = nltk.word_tokenize(data) # data is a list now
    return data
    
# print each individual song from list
for filename in songlist :
    if not filename.endswith(".txt") :
        continue 

    with open(os.path.join(countrysongs,filename),"r") as file_handler :
        data = file_handler.read()
        print("\n" + filename)
        data = clean(data)
        print("\n", data)

# 

