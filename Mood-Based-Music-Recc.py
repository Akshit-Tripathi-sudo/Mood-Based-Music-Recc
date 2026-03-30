import random
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = {
    "text": [

        # HAPPY
        "I feel amazing today","I am so happy","Life is beautiful","I feel great",
        "Everything is awesome","I am smiling","I feel fantastic","Best day ever",

        # SAD 
        "I am very sad","I feel depressed","I want to cry","Life is painful",
        "I feel empty","I am heartbroken","Feeling low","I feel broken",

        # ANGRY
        "I am angry","This is frustrating","I feel rage","I hate everything",
        "I am irritated","I am mad","I feel aggressive","This is annoying",

        # CHILL
        "I feel calm","Just relaxing","Peaceful vibes","Feeling chill",
        "I feel relaxed","No stress","Everything is peaceful","I feel light",

        # ROMANTIC
        "I am in love","Feeling romantic","I miss someone","Thinking about crush",
        "Love is in the air","I feel attached","I want a hug","I feel प्यार",

        # ENERGETIC 
        "I want to dance","party mood","full energy","i am hyped",
        "i feel excited","lets party hard","i feel energetic",
        "i am super pumped",

        # MOTIVATED
        "I want to succeed","I am focused on goals","I feel motivated",
        "I will achieve something","Time to work hard","No excuses",
        "I am determined","I want success",

        # LONELY 
        "I feel alone","no one is with me","i feel isolated",
        "i am by myself","i have nobody","i feel left out",
        "i feel disconnected","i am lonely",

        # PARTY
        "Let's party","I want to enjoy","Club night","Dance all night",
        "Let's celebrate","Party mood","Turn up the music","Let's have fun",

        # FOCUS
        "I need to study","I want to concentrate","Focus mode",
        "No distractions","I want to work","Deep work time",
        "I need productivity","Time to study hard",

        # NORMAL
        "I feel okay","Just normal","Nothing special today",
        "I am fine","Just another day","I feel neutral",
        "Everything is normal","I feel average"
    ],

    "mood": [
        "happy","happy","happy","happy","happy","happy","happy","happy",
        "sad","sad","sad","sad","sad","sad","sad","sad",
        "angry","angry","angry","angry","angry","angry","angry","angry",
        "chill","chill","chill","chill","chill","chill","chill","chill",
        "romantic","romantic","romantic","romantic","romantic","romantic","romantic","romantic",
        "energetic","energetic","energetic","energetic","energetic","energetic","energetic","energetic",
        "motivated","motivated","motivated","motivated","motivated","motivated","motivated","motivated",
        "lonely","lonely","lonely","lonely","lonely","lonely","lonely","lonely",
        "party","party","party","party","party","party","party","party",
        "focus","focus","focus","focus","focus","focus","focus","focus",
        "normal","normal","normal","normal","normal","normal","normal","normal"
    ]
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df["text"])
y = df["mood"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


songs = {
    "happy": ["Happy","On Top of the World","Good Life","Ilahi","Kala Chashma"],
    "sad": ["Someone Like You","Fix You","Let Her Go","Channa Mereya","Tadap Tadap"],
    "angry": ["Numb","Stronger","Lose Yourself","Zinda","Sultan"],
    "chill": ["Sunflower","Perfect","Kasoor","Kho Gaye Hum Kahan","Sham"],
    "romantic": ["Tum Hi Ho","Raabta","Kesariya","Shayad","Pehla Nasha"],
    "energetic": ["Lollipop Lagelu","Kamariya Kare Lapalap","Malhari","Badtameez Dil"],
    "motivated": ["Hall of Fame","Lakshya","Apna Time Aayega","Kar Har Maidaan Fateh"],
    "lonely": ["Someone You Loved","Phir Le Aya Dil","Hasi Ban Gaye","Channa Mereya"],
    "party": ["DJ Waley Babu","Abhi Toh Party Shuru Hui Hai","Kala Chashma, raja ji ke dilwa"],
    "focus": ["Lo-fi Beats","Interstellar Theme","Instrumental Study, Lakshya","Time","Sham"],
}


all_songs = []
for s in songs.values():
    all_songs.extend(s)

songs["normal"] = list(set(all_songs))


def predict_mood(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]


def recommend_song(mood):
    return random.choice(songs.get(mood, songs["normal"]))



print("\n Mood-Based Song Recommender")

user_input = input("How are you feeling today? -> ")

predicted_mood = predict_mood(user_input)
song = recommend_song(predicted_mood)

print(f"\n Detected Mood: {predicted_mood}")
print(f" Recommended Song: {song}")
