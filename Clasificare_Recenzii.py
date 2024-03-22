import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix

data = pd.read_csv('IMDB Dataset.csv')
print()
print("Formatul tabelului de recenzii:", data.shape)
print("Afisez primele recenzii:")
print(data.head())

print()
print(data.info())

print()
print("Numar recenzii positive si numar recenzii negative:")
print(data.sentiment.value_counts())

data.sentiment.replace('positive', 1, inplace=True)
data.sentiment.replace('negative', 0, inplace=True)
print()
print("Transform sentimentele in 1 (pozitiv) si 0 (negativ):")
print(data.head(10))

print()
print("Datele inainte sa fie curatate:")
print(data.review[0])

# Curat datele de taguri HTML
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

data.review = data.review.apply(clean)
print()
print("1. Datele curatate de taguri HTML:")
print(data.review[0])

# Curat datele de caractere speciale
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

data.review = data.review.apply(is_special)
print()
print("2. Datele curatate de caractere speciale:")
print(data.review[0])

# Convertesc totul in lowercase
def to_lower(text):
    return text.lower()

data.review = data.review.apply(to_lower)
print()
print("3. Datele convertite in lowercase:")
print(data.review[0])

# Elimin cuvintele de oprire (the, and, is, to, for etc.)
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

data.review = data.review.apply(rem_stopwords)
print()
print("4. Datele fara cuvinte de oprire:")
print(data.review[0])

# Elimin sufixele si alte componente care pot varia (stemming)
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(stem_txt)
print()
print("5. Datele fara sufixe:")
print(data.review[0])

print()
print(data.head())

# Crearea modelului

# Crearea unui Bag Of Words
X = np.array(data.iloc[:, 0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(data.review).toarray()
print()
print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

print()
print(X)

# Impartirea datelor in seturi de antrenare, validare si testare
# x - matricea de caracteristici (bag of words) obtinuta din recenziile de filme
# Y - vectorul de etichete (sentiment) asociat recenziilor
# Datele de antrenare sunt folosite sa antreneze modelul, datele de validare pentru ajustarea hiperparametrilor,
# iar datele de testare sunt folosite pentru a evalua performantele acestuia
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=9)
trainx, valx, trainy, valy = train_test_split(trainx, trainy, test_size=0.25, random_state=9)

print()
print("Train shapes : X = {}, y = {}".format(trainx.shape, trainy.shape))
print("Validation shapes : X = {}, y = {}".format(valx.shape, valy.shape))
print("Test shapes : X = {}, y = {}".format(testx.shape, testy.shape))

# Definirea si antrenarea modelelor
# Creez 3 instante ale clasificatorilor bayesieni naivi cu tipurile specifice:
# GaussianNB - pentru date cu distributie gaussiana
# MultinomialNB - pentru date distributie multinomiala
# BernoulliNB - pentru date binare
gnb, mnb, bnb = GaussianNB(), MultinomialNB(alpha=1.0, fit_prior=True), BernoulliNB(alpha=1.0, fit_prior=True)
gnb.fit(trainx, trainy)
mnb.fit(trainx, trainy)
bnb.fit(trainx, trainy)

models = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']

# Predicitii si acuratete pentru a alege cel mai potrivit clasificator:
# Se fac predictii pentru fiecare clasificator pe baza setului de date de validare
ypg_val = gnb.predict(valx)
ypm_val = mnb.predict(valx)
ypb_val = bnb.predict(valx)
print()
# accuracy_score compara rezultatele predictiilor cu etichetele reale din setul de validare
print("Acuratete validare - Gaussian = ", accuracy_score(valy, ypg_val))
print("Acuratete validare - Multinomial = ", accuracy_score(valy, ypm_val))
print("Acuratete validare - Bernoulli = ", accuracy_score(valy, ypb_val))

accuracy_val_scores_test = [
    accuracy_score(valy, ypg_val),
    accuracy_score(valy, ypm_val),
    accuracy_score(valy, ypb_val)
]

plt.bar(models, accuracy_val_scores_test, color=['blue','purple', 'pink'])
plt.xlabel('Algoritmi')
plt.ylabel('Acuratețe')
plt.title('Compararea acurateții între GaussianNB, MultinomialNB si BernoulliNB\npe baza valorilor de validare')
plt.ylim(0, 1)
plt.show()

# Se fac predictii pentru fiecare clasificator pe baza setului de date de testare
ypg_test = gnb.predict(testx)
ypm_test = mnb.predict(testx)
ypb_test = bnb.predict(testx)
print()
# accuracy_score compara rezultatele predictiilor cu etichetele reale din setul de testare
print("Acuratete testare - Gaussian = ", accuracy_score(testy, ypg_test))
print("Acuratete testare - Multinomial = ", accuracy_score(testy, ypm_test))
print("Acuratete testare - Bernoulli = ", accuracy_score(testy, ypb_test))

accuracy_test_scores_test = [
    accuracy_score(testy, ypg_test),
    accuracy_score(testy, ypm_test),
    accuracy_score(testy, ypb_test)
]

plt.bar(models, accuracy_val_scores_test, color=['skyblue','purple', 'magenta'])
plt.xlabel('Algoritmi')
plt.ylabel('Acuratețe')
plt.title('Compararea acurateții între GaussianNB, MultinomialNB si BernoulliNB\npe baza valorilor de testare')
plt.ylim(0, 1)
plt.show()

# Salvez modelul antrenat (bnb) intr-un fisier utilizand modul de serializare pickle
pickle.dump(bnb, open('model1.pkl', 'wb'))
pickle.dump(mnb, open('model1.pkl', 'wb'))
pickle.dump(gnb, open('model1.pkl', 'wb'))

# Salvare vocabular Bag Of Words
word_dict = cv.vocabulary_
pickle.dump(word_dict, open('bow.pkl', 'wb'))


reviews = [
    """Terrible. Complete trash. Brainless tripe. Insulting to anyone who isn't an 8 year old fan boy. Im actually
    pretty disgusted that this movie is making the money it is - what does it say about the people who brainlessly
    hand over the hard earned cash to be 'entertained' in this fashion and then come here to leave a positive 8.8 review??
    Oh yes, they are morons. Its the only sensible conclusion to draw. How anyone can rate this movie amongst the pantheon
    of great titles is beyond me. So trying to find something constructive to say about this title is hard...I enjoyed Iron Man?
    Tony Stark is an inspirational character in his own movies but here he is a pale shadow of that...About the only 'hook'
    this movie had into me was wondering when and if Iron Man would knock Captain America out...Oh how I wished he had :(
    What were these other characters anyways? Useless, bickering idiots who really couldn't organise happy times in a brewery.
    The film was a chaotic mish mash of action elements and failed 'set pieces'... I found the villain to be quite amusing.
    And now I give up. This movie is not robbing any more of my time but I felt I ought to contribute to restoring the obvious
    fake rating and reviews this movie has been getting on IMDb.""",
    
    """Absolutely dreadful! A complete waste of time. The plot was incoherent, the acting was wooden, and the special effects were laughably bad. 
    I can't believe I sat through the entire thing. Save yourself the agony and steer clear of this cinematic disaster.""",
    
    """Terrible from start to finish! The storyline was confusing, the acting was painfully awkward, and the dialogue felt like 
    it was written by a child. I couldn't wait for it to end. It's astonishing how a film this bad even got made. 
    Save your money and skip this cinematic catastrophe.""",
    
    """Dreadful film, an absolute letdown. The plot was so predictable, and the characters were one-dimensional. 
    The acting was cringe-worthy, and I found myself checking my watch throughout, counting down the minutes until it was over. 
    A complete disappointment that I wouldn't recommend to anyone.""",
    
    """Awful movie, no redeeming qualities whatsoever. The storyline was uninspired, the performances were lackluster, 
    and the whole production felt like a low-budget disaster. I couldn't connect with any of the characters, and the 
    so-called 'twist' at the end was just plain absurd. A regrettable cinematic experience that I wouldn't wish on my worst enemy""",
    
    """Absolutely outstanding! This film surpassed all expectations and delivered an unforgettable cinematic experience. 
    The plot was brilliantly crafted, keeping me on the edge of my seat from beginning to end. 
    The performances were exceptional, breathing life into characters I genuinely cared about. 
    Visually stunning with breathtaking cinematography, and the soundtrack perfectly complemented the emotional journey. 
    A true masterpiece that deserves every bit of praise it receives. Highly recommended for anyone seeking a truly remarkable film.""",
    
    """Exquisite film that left me in awe! From the captivating storyline to the superb acting, every element came together seamlessly. 
    The character development was exceptional, making me feel deeply connected to their journeys. 
    The cinematography was breathtaking, and the musical score heightened the emotional impact. This is not just a movie; it's a work of art. 
    I can't recommend it enough to anyone who appreciates a film that transcends expectations and lingers in your thoughts long after 
    the credits roll.""",
    
    """Fantastic movie! Gripping plot, stellar performances, and visually stunning. A must-see.""",
    
    """Outstanding! This film exceeded all expectations. Compelling story, brilliant acting, and visually stunning. A cinematic gem!""",
    
    """Incredible film! Compelling story, brilliant performances, and visually stunning cinematography. A true masterpiece!"""
]

# Etichetele recenziilor
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

results = []

for i in range(10):
    rev = reviews[i]
    f1 = clean(rev)
    f2 = is_special(f1)
    f3 = to_lower(f2)
    f4 = rem_stopwords(f3)
    f5 = stem_txt(f4)

    bow, words = [], word_tokenize(f5)
    for word in words:
        bow.append(words.count(word))

    inp = []
    for j in word_dict:
        inp.append(f5.count(j[0]))

    # Se realizeaza predicii pentru fiecare algoritm
    y_pred_bnb = bnb.predict(np.array(inp).reshape(1, 1000))
    y_pred_mnb = mnb.predict(np.array(inp).reshape(1, 1000))
    y_pred_gnb = gnb.predict(np.array(inp).reshape(1, 1000))

    # Se adauga rezultatele in lista de rezultate
    results.append((y_pred_bnb[0], y_pred_mnb[0], y_pred_gnb[0]))

# Lista de rezultate este transformata intr-un array NumPy
results_array = np.array(results)

# Calcularea si afisarea matricelor de confuzie pentru fiecare algoritm
confusion_matrix_bnb = confusion_matrix(y_true, results_array[:, 0])
confusion_matrix_mnb = confusion_matrix(y_true, results_array[:, 1])
confusion_matrix_gnb = confusion_matrix(y_true, results_array[:, 2])

print("Matricea de confuzie pentru BernoulliNB:")
print(confusion_matrix_bnb)

print("\nMatricea de confuzie pentru MultinomialNB:")
print(confusion_matrix_mnb)

print("\nMatricea de confuzie pentru GaussianNB:")
print(confusion_matrix_gnb)