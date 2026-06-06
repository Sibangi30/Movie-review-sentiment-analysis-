#Data Cleaning
#null values
print(df.isnull().sum())
#missing values
missing=df.isnull().sum()
missing[missing>0]
df.drop(columns=['class','text'])
#Converting text to number
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])
#finding duplicates
print(df.duplicated().sum())
print(df.shape)
print(df.head())
print(df['class'].value_counts())

# Number of characters
df['num_characters'] = df['text'].astype(str).apply(len)

# Number of words
df['num_words'] = df['text'].astype(str).apply(lambda x: len(word_tokenize(x)))

# Number of sentences
df['num_sentences'] = df['text'].astype(str).apply(lambda x: len(sent_tokenize(x)))

print(df.head())

# Summary all reviews
print(df[['num_characters','num_words','num_sentences']].describe())

# Positive reviews only
df[df['class'] == 1][['num_characters','num_words','num_sentences']].describe()

# Negative reviews only
df[df['class'] == 0][['num_characters','num_words','num_sentences']].describe()

#Data preprocessing
import pandas as pd
import nltk
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define abbreviation dictionary
abbreviations = {
    "i'm": "i am",
    "can't": "cannot",
    "btw": "by the way",
    "won't": "will not",
    "don't": "do not",
    "tq": "thank you",
    "it's": "it is",
    "that's": "that is",
    "idk": "i do not know",
    "imo": "in my opinion",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "asap": "as soon as possible",
    "eta": "estimated time of arrival",
    "faq": "frequently asked questions",
    "dm": "direct message",
    "pm": "private message",
    "msg": "message",
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "gr8": "great",
    "b4": "before",
    "bday": "birthday",
    "g2g": "got to go",
    "jk": "just kidding",
    "np": "no problem",
    "ic": "i see",
    "nvm": "never mind",
    "sci-fi": "science fiction",
    "rom-com": "romantic comedy",
    "docu": "documentary",
    "bio-pic": "biographical picture",
    "anime": "animation",
    "cgi": "computer generated imagery",
    "vfx": "visual effects",
    "sfx": "special effects",
    "fx": "effects",
    "pov": "point of view",
    "dp": "director of photography",
    "ost": "original soundtrack",
    "bgm": "background music",
    "imax": "image maximum format",
    "pg": "parental guidance",
    "pg-13": "parents strongly cautioned",
    "r": "restricted",
    "nc-17": "no children under 17",
    "u": "universal",
    "ua": "universal adult",
    "imdb": "internet movie database",
    "ott": "over the top streaming",
    "bo": "box office",
    "mcu": "marvel cinematic universe",
    "dceu": "dc extended universe",
    "lotr": "lord of the rings",
    "got": "game of thrones",
    "hp": "harry potter",
    "sw": "star wars",
    "trek": "star trek",
    "spoiler": "plot reveal",
    "cliche": "overused trope",
    "meta": "self-referential",
    "camp": "exaggerated style",
    "cult": "cult classic",
    "indie": "independent film",
    "netflix": "netflix",
    "prime": "amazon prime video",
    "disney+": "disney plus",
    "hbo": "home box office",
    "hulu": "hulu",
    "apple tv+": "apple tv plus"
}

# Create the pattern using the defined abbreviations dictionary
sorted_keys = sorted(abbreviations.keys(), key=len, reverse=True)
pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in sorted_keys) + r')\b')

def preprocessing_text(text):
    text = str(text).lower()
    # Remove HTML-like tags first
    text = re.sub(r'<.*?>', '', text)

    # Expand abbreviations using the pre-compiled pattern
    text = pattern.sub(lambda x: abbreviations[x.group()], text)

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]','',text)
    # Normalize whitespace: replace multiple spaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)