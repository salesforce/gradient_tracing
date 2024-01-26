'''
Unites the per-topic files into a single json
'''

import json

full_collection = {}
topics = [
    "famous actors", "famous authors", "famous scientists", "famous philosophers",
    "famous politicians", "famous musicians", "famous explorers", "famous CEOs",
    "famous inventors", "historical figures",
    "ancient Greece", "the Roman Empire", "the Middle Ages", "the Renaissance",
    "the Industrial Revolution", "World War I", "World War II", "the Cold War",
    "the French Revolution", "the American Civil War",
    "classical composers", "rock bands", "jazz artists", "musical instruments",
    "pop music", "operas", "musicals", "music albums", "music genres", "national anthems",
    "Marvel Cinematic Universe", "famous directors", "film genres", "film awards",
    "silent films", "horror films", "film studios", "action films", "sci-fi films", "Hollywood",
    "sitcoms", "news broadcasts", "documentaries", "TV networks", "sci-fi TV shows",
    "famous TV anchors", "reality TV", "CSI (TV show)", "game shows", "animated TV shows",
    "Olympics", "soccer", "tennis", "basketball", "baseball", "stadiums",
    "golf", "boxing", "ice hockey", "rugby",
    "mountains", "rivers", "oceans", "deserts", "forests", "storms", "seas",
    "islands", "valleys", "volcanoes",
    "plants", "mammals", "birds", "reptiles", "amphibians", "insects", "fish",
    "planets", "stars", "galaxies", "moons", "asteroids", "comets",
    "black holes", "satellites", "observatories",
    "thermodynamics", "optics", "quantum mechanics", "relativity theory", "nuclear physics",
    "political parties",
    "diseases", "medical treatments", "human anatomy", "medical symptoms",
    "meteorology", "storms", "earthquakes",
    "famous architects", "construction", "historical buildings", "modern buildings", "urban planning",
    "New York", "Tokyo", "London", "Paris", "Mumbai", "Shanghai", "Sao Paulo",
    "Los Angeles", "Moscow", "Sydney", "Berlin", "Rome", "Bangkok", "Toronto",
    "Istanbul", "Singapore", "Cairo",
    "languages", "religions", "currencies", "Fortune 500 companies", "Vikings",
    "toys", "wine", "coffee", "desserts", "banks", "hotels", "trains", "cars",
    "aviation", "ships", "genetics", "Mars", "dinosaurs", "telescopes", "famous monarchs",
    "theater", "viruses", "vaccines",
    "USA", "China", "India", "Brazil", "France", "Russia", "Australia", "Canada",
    "Japan", "Germany", "Spain", "Italy", "Mexico", "South Korea", "Argentina", "Israel",
    "Africa", "Europe", "Asia", "North America", "South America", "Australia", "Antarctica"
]
topics += ['furniture', 'famous bridges', 'famous skyscrapers', 'trees', 'chemistry', 'airlines', 'ancient battles', 'computer brands',
          'famous philanthropists', 'famous paintings', 'famous sculptures', 'famous churches']
topics += ['metals', 'stand-up comedy', 'holidays', 'horses', 'clergy']
total_num_false = 0
total_num_true = 0
total_num = 0
for topic in topics:
  with open('open_ai_output_' + topic + '.json', 'r') as f:
      full_collection[topic] = json.load(f)
      for entry in full_collection[topic]:
          total_num += 1
          if entry['truth_value']:
              total_num_true += 1
          else:
              total_num_false += 1

print(total_num, total_num_true, total_num_false)

with open('../fact.json', 'w') as f:
    json.dump(full_collection, f, indent=4)










