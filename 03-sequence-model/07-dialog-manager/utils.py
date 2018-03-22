import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'models/intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'models/tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'models/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'models/thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'models/word_embeddings.tsv',
}

def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    import pandas as pd

    embeddings = pd.read_csv(embeddings_path, sep='\t', header=None)

    N, D = embeddings.shape

    embeddings = {
        k:v for k, v in zip(
            embeddings.iloc[:,0].values,
            embeddings.iloc[:,1:].values
        )
    }
    embeddings_dim = D - 1

    return (embeddings, embeddings_dim)

def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    # filter out out-of-vocabulary words:
    existing_words = [word for word in question.split() if word in embeddings]

    # if all out-of-vocabulary words, return default embedding:
    if not existing_words:
        return np.zeros(dim)

    word_embeddings = np.asarray(
        [embeddings[word] for word in existing_words]
    )

    # question embedding as mean of word embeddings:
    question_embedding = np.mean(word_embeddings, axis = 0)

    return question_embedding

def get_best_answer_id(question_embedding, answer_embeddings):
    """
        question_embedding: question embedding
        answer_embeddings: answer embeddings

        result: best answer id
    """
    from sklearn.metrics.pairwise import cosine_similarity

    scores = cosine_similarity(
        question_embedding.reshape(1, -1),
        answer_embeddings
    ).ravel()

    # best answer id:
    best_id, _ = max(enumerate(scores), key = lambda t: t[1])

    return best_id

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
