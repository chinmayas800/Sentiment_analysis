from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import mglearn



reviews_train = load_files("data/aclImdb/train/")
load_files returns a bunch, containing training texts and trai
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = load_files("data/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
vect = CountVectorizer()
vect.fit(text_train)
bag_of_words = vect.transform(bards_words)#converting to sparse matrix form
feature_names = vect.get_feature_names()


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: ", grid.best_score_)
print("Best parameters: ", grid.best_params_)


vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
feature_names = vect.get_feature_names()

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: ", grid.best_score_)

#using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None), LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: ", grid.best_score_)


vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset:
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over dataset:
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("features with lowest tfidf")
print(feature_names[sorted_by_tfidf[:20]])
print("features with highest tfidf")
print(feature_names[sorted_by_tfidf[-20:]])


mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps["logisticregression"].coef_,
 feature_names, n_top_features=40)
plt.title("tfidf-coefficient")


cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print(len(cv.vocabulary_))
print(cv.get_feature_names())

# extract scores from grid_search
scores = [s.mean_validation_score for s in grid.grid_scores_]
scores = np.array(scores).reshape(-1, 3).T
# visualize heatmap
heatmap = mglearn.tools.heatmap(scores, xlabel="C", ylabel="ngram_range",
 xticklabels=param_grid['logisticregression__C'],
 yticklabels=param_grid['tfidfvectorizer__ngram_range'],
 cmap="viridis", fmt="%.3f")
plt.colorbar(heatmap);


general principles here.
import spacy
import nltk
# load spacy's English language models
en_nlp = spacy.load('en')
# instantiate NLTK's Porter stemmer
stemmer = nltk.stem.PorterStemmer()
# define function to compare lemmatization in spacy with stemming in NLKT
def compare_normalization(doc):
	# tokenize document in spacy:
 doc_spacy = en_nlp(doc)
 # print lemmas found by spacy
 print("Lemmatization:")
 print([token.lemma_ for token in doc_spacy])
 # print tokens found by Porter stemmer
 print("Stemming:")
 print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])


# Technicallity: we want to use the regexp based tokenizer that is used by CountVectorizer
# and only use the lemmatization from SpaCy. To this end, we replace en_nlp.tokenizer (the SpaCy tokenizer)
# with the regexp based tokenization
import re
# regexp used in CountVectorizer:
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model and save old tokenizer
en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the regexp above
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))
# create a custom tokenizer using the SpaCy document processing pipeline
# (now using our own tokenizer)
Rescaling the data with TFIDF | 327
def custom_tokenizer(document):
 doc_spacy = en_nlp(document, entity=False, parse=False)
 return [token.lemma_ for token in doc_spacy]
# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
Let’s transform the data and inspect the vocabulary size:
# transform text_train using CountVectorizer with lemmatization
X_train_lemma = lemma_vect.fit_transform(text_train)
print("X_train_lemma.shape: ", X_train_lemma.shape)
# Standard CountVectorizer for reference
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train.shape: ", X_train.shape)
