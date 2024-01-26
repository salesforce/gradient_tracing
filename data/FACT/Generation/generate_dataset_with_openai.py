'''
repeatedly calls chatgpt to create fact
'''

import openai
import json
import time

temp = 0.7

key = ''

openai.api_key = key

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
    "Africa", "Europe", "Asia", "North America", "South America", "Australia", "Antarctica",
    'furniture', 'famous bridges', 'famous skyscrapers', 'trees', 'chemistry', 'airlines', 'ancient battles',
    'computer brands', 'famous philanthropists', 'famous paintings', 'famous sculptures', 'famous churches',
    'metals', 'stand-up comedy', 'holidays', 'horses', 'clergy'
]


example_json_file = 'example_zoology.json'

with open(example_json_file, 'r') as f:
  example_str = json.dumps(json.load(f))

for topic in topics:
  start = time.time()
  print('start', start, 'topic', topic)
  prompt = "Provide a list of 6 statements on the topic of {}, some true and some false. For each statement, provide two different rephrases. For each main term in the original statement, provide four additional statements about the term, two true and two false; all four statements should be unrelated to the original statement. For every statement (original, rephrases, and term-statements), provide its negation. Label all statements as true or false. The statements should be objective and unambiguous, and all information used must be well-known to the layperson. The statements should not include numbers, dates or anastrophes. Structure the output as a json file, and include nothing except the json file. See below for examples on the different topic of zoology.\n".format(topic)
  prompt += example_str
  error = True
  error_counter = 0
  while error:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        # max_tokens=1000,
        # logit_bias={"100257": -100}
      )
      break
    except:
      error_counter += 1
      print('Error number', error_counter)
      continue
  with open('open_ai_output_str_' + topic + '.json', 'w') as f:
    json.dump(response["choices"][0]["message"]["content"], f)
  try:
    with open('open_ai_output_str_' + topic + '.json', 'r') as f:
        json_as_string = json.load(f)
        json_as_dict = json.loads(json_as_string)
    with open('open_ai_output_' + topic + '.json', 'w') as f:
        json.dump(json_as_dict, f)
  except:
    print('topic', topic, 'conversion to json failed!')
  end = time.time()
  print('end', end, 'topic', topic)
  print('time taken', end - start, 'errors', error_counter)









