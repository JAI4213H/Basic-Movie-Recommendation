from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data = {
    "Interstellar": "space exploration sci-fi time dilation black hole",
    "Inception": "dream mind bending sci-fi thriller subconscious",
    "The Dark Knight": "batman crime action joker thriller",
    "Avengers Endgame": "superhero marvel action time travel",
    "Iron Man": "technology superhero marvel action genius",
    "Doctor Strange": "magic marvel multiverse fantasy action",
    "The Matrix": "simulation sci-fi action artificial intelligence",
    "John Wick": "assassin action revenge gunfight thriller",
    "Gladiator": "roman empire war action revenge history",
    "Titanic": "romance tragedy ship historical drama",
    "The Notebook": "romance emotional love drama",
    "La La Land": "music romance dream drama",
    "Whiplash": "music intensity drama motivation",
    "Fight Club": "psychological drama identity rebellion",
    "Forrest Gump": "life journey emotional drama history",
    "The Shawshank Redemption": "prison hope friendship drama",
    "The Godfather": "mafia crime family drama",
    "The Godfather Part II": "mafia crime power legacy drama",
    "Pulp Fiction": "crime nonlinear story dark humor",
    "Se7en": "crime thriller serial killer mystery",
    "Zodiac": "crime investigation mystery serial killer",
    "Gone Girl": "psychological thriller mystery marriage",
    "Shutter Island": "psychological thriller mystery mental illness",
    "The Prestige": "magic rivalry mystery thriller",
    "Memento": "memory loss thriller nonlinear mystery",
    "Dune": "desert sci-fi politics war space",
    "Blade Runner 2049": "future sci-fi dystopia artificial intelligence",
    "Avatar": "alien world sci-fi adventure nature",
    "Gravity": "space survival thriller isolation",
    "The Martian": "space survival science astronaut",
    "Star Wars": "space opera sci-fi adventure jedi",
    "Star Trek": "space exploration sci-fi crew adventure",
    "Jurassic Park": "dinosaurs science adventure thriller",
    "Jaws": "shark thriller ocean survival",
    "The Lion King": "animation family adventure kingdom",
    "Frozen": "animation magic disney adventure",
    "Coco": "animation family music emotional",
    "Toy Story": "animation toys friendship adventure",
    "Up": "animation emotional adventure old man",
    "Finding Nemo": "animation ocean adventure family",
    "The Incredibles": "animation superhero family action",
    "Spider-Man": "superhero marvel action teen",
    "Batman Begins": "batman origin superhero action",
    "Superman": "superhero alien action hero",
    "Deadpool": "superhero comedy action adult humor",
    "Logan": "superhero emotional action dark",
    "Black Panther": "superhero africa marvel action",
    "Thor": "superhero mythological action marvel",
    "Captain America": "superhero war patriot action",
    "Hulk": "superhero strength monster action",
    "Transformers": "robots action sci-fi war",
    "Pacific Rim": "giant robots monsters action",
    "Mad Max Fury Road": "post apocalyptic action survival",
    "The Hunger Games": "survival dystopia action youth",
    "Harry Potter": "magic wizard school fantasy adventure",
    "Lord of the Rings": "fantasy epic adventure ring war",
    "The Hobbit": "fantasy adventure dragon quest",
    "Pirates of the Caribbean": "pirates adventure ocean fantasy",
    "Indiana Jones": "adventure treasure archaeology action",
    "National Treasure": "treasure hunt mystery adventure",
    "Sherlock Holmes": "detective mystery investigation crime",
    "Mission Impossible": "spy action thriller espionage",
    "James Bond": "spy action agent thriller",
    "Kingsman": "spy action comedy secret agent",
    "The Bourne Identity": "spy memory action thriller",
    "Fast and Furious": "cars racing action family",
    "Need for Speed": "cars racing action speed",
    "Rush": "racing cars biography drama",
    "Ford v Ferrari": "cars racing history drama",
    "Rocky": "boxing sports motivation",
    "Creed": "boxing sports legacy drama",
    "The Karate Kid": "martial arts sports training",
    "Ip Man": "martial arts biography action",
    "Enter the Dragon": "martial arts action classic",
    "Kung Fu Panda": "animation martial arts comedy",
    "Rush Hour": "comedy action police",
    "The Hangover": "comedy friends party chaos",
    "Superbad": "teen comedy friendship",
    "21 Jump Street": "comedy police undercover",
    "The Mask": "comedy fantasy transformation",
    "Ace Ventura": "comedy detective animals",
    "Mr Bean": "comedy silent humor",
    "Home Alone": "comedy family christmas",
    "Die Hard": "action christmas thriller",
    "The Conjuring": "horror paranormal ghost",
    "Insidious": "horror supernatural fear",
    "Annabelle": "horror doll paranormal",
    "The Nun": "horror church supernatural",
    "It": "horror clown fear",
    "Hereditary": "horror psychological family",
    "A Quiet Place": "horror survival silence",
    "Get Out": "horror social thriller",
    "Us": "horror psychological thriller",
    "The Exorcist": "horror possession classic",
    "Scream": "horror slasher teen",
    "Saw": "horror torture thriller",
    "Final Destination": "horror fate death",
    "Parasite": "thriller social class drama",
    "Joker": "psychological character study crime",
    "Taxi Driver": "psychological crime loneliness",
    "The Wolf of Wall Street": "finance greed biography",
    "Moneyball": "sports analytics baseball",
    "The Social Network": "technology facebook drama",
    "Steve Jobs": "technology biography innovation",
    "A Beautiful Mind": "math genius biography drama",
    "The Imitation Game": "codebreaking war intelligence",
    "Oppenheimer": "physics war biography nuclear",
    "Interstellar 2": "space future sci-fi exploration hypothetical"
}

movies = list(movie_data.keys())
features = list(movie_data.values())

cv = CountVectorizer()
vector = cv.fit_transform(features)

similarity = cosine_similarity(vector)

desire = movies.index(str(input("Tell movie Name: ")))

match = list(enumerate(similarity[desire]))
match = sorted(match, key=lambda x: x[1], reverse=True) 

print(f"The top 5 Best matches for the movie {movies[desire]} is")
for i in match[1:6]:
    print(movies[i[0]],",   with score", i[1] * 100)
    





















