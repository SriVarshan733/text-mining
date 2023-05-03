#SOURCE CODE FOR TEXT MINING FOR OUTPUT VISIT .ipynd file

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import sent_tokenize
text = '''Hello Mr. Jones, how are you doing today? The weather is great, and city is awesome.
The sky is bright-blue. You should't call for meeting today'''
tokenized_text = sent_tokenize(text)
print(tokenized_text)

from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print(tokenized_word)

from nltk.probability import FreqDist
frequency = FreqDist(tokenized_word)
print(frequency)

frequency.most_common(3)

import matplotlib.pyplot as plt
frequency.plot(30, cumulative=False)
plt.show()

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

filtered_sent = []
for w in tokenized_text:
  if w not in stop_words:
    filtered_sent.append(w)
print("Tokenized Sentence: ", tokenized_text)
print("Filtered Sentence: ", filtered_sent)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
stemmed_words=[]

for w in filtered_sent:
  stemmed_words.append(ps.stem(w))

print("Filtered Sentence:", filtered_sent)
print("Stemmed Sentence:", stemmed_words)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "Working"
print("Lemmatized Word: ", lem.lemmatize(word, "v"))
print("Stemmed Word: ", stem.stem(word))

word = "Flying"
print("Lemmatized Word: ", lem.lemmatize(word, "v"))
print("Stemmed Word: ", stem.stem(word))

sentence = "Albert Einstein was born in Ulm, Germany in 1879"
tokens = nltk.word_tokenize(sentence)
print(tokens)

nltk.pos_tag(tokens)