from pathlib import Path
import pickle
from typing import List

import nltk
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from rs.rnn2 import LoadedModel


def export_context_vectors(model: LoadedModel, word: str,
                           pos='nouns', corpus='RuTenTen'):
    import rl_wsd_labeled
    ctx = rl_wsd_labeled.get_contexts(
        rl_wsd_labeled.contexts_filename(pos, corpus, word))
    contexts = [(tokenize(left), tokenize(right))
                for (left, _, right), _ in ctx[1]]
    cv = model.contexts_vectors(contexts)
    pred = model.predictions(contexts)
    probs = model.session.run(
        model.softmax, feed_dict=model.feed_dict(contexts))
    words = [preprocess(word) for _, word, _ in ctx[1]]
    labels = np.arange(probs.shape[1])
    losses = [log_loss(y_true=[model.vocabulary.get_id(w)],
                       y_pred=probs[i:i + 1],
                       labels=labels)
              for i, w in enumerate(words)]
    with open('cv-{}.pkl'.format(word), 'wb') as f:
        pickle.dump({
            'word': word,
            'corpus': corpus,
            'POS': pos,
            'cv': cv,
            'pred': pred,
            'ctx': ctx,
            'contexts': contexts,
            'losses': losses,
        }, f)


def preprocess(s: str) -> str:
    return s.lower().replace('"', '')


def tokenize(s: str) -> List[str]:
    return nltk.wordpunct_tokenize(preprocess(s))


def save_embeddings(cv_path: Path, output: Path):
    with cv_path.open('rb') as f:
        data = pickle.load(f)

    vectors = []
    labels = [['sense', 'context']]
    for ((left, word, right), sense), vector in zip(data['ctx'][1], data['cv']):
        labels.append([sense, ' '.join([left, word, right])])
        vectors.append(vector)

    output.mkdir(exist_ok=True)
    labels_path = output.joinpath('labels.tsv')
    labels_path.write_text(
        '\n'.join('\t'.join(row) for row in labels))
    vectors = np.array(vectors)

    with tf.Session() as session:
        embedding_var = tf.Variable(vectors, trainable=False,
                                    name='vectors')
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(session, str(output.joinpath('model.ckpt')))
        summary_writer = tf.summary.FileWriter(str(output))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = str(labels_path)
        projector.visualize_embeddings(summary_writer, config)


def evaluate_clf(cv_path: Path):
    with cv_path.open('rb') as f:
        data = pickle.load(f)
    all_xs = data['cv']
    all_ys = np.array([int(sense) - 1 for _, sense in data['ctx'][1]])
    all_xs, all_ys = shuffle(all_xs, all_ys)

    def eval_clf(clf):
        scores = cross_val_score(clf, all_xs, all_ys, cv=10)
        print('Accuracy: {:.3f} ± {:.3f}'.format(
            np.mean(scores), 2 * np.std(scores)))

    for clf in [
            KNeighborsClassifier(
                metric='cosine', algorithm='brute', n_neighbors=3),
            NearestCentroid(metric='cosine'),
            ]:
        eval_clf(clf)
