#!/usr/bin/env python
from __future__ import print_function
import os.path
import argparse
import random
import json
import csv
import sys
from operator import itemgetter
from collections import Counter

from rs.utils import avg
from rlwsd.utils import mystem
from rs.active_dict.loader import get_ad_word
from rs.supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SupervisedWrapper, sorted_senses, get_accuracy_estimate, get_mfs_baseline, \
    show_tsne
from rs.cluster import get_context_vectors
from rs import cluster_methods, supervised


def train_model(word, ad_word_data, ad_root, method=None, **model_params):
    weights = None if model_params.pop('no_weights', None) else load_weights(
        word, root=ad_root, lemmatize=model_params.get('lemmatize'))
    train_data = get_ad_train_data(word, ad_word_data)
    model = None
    if train_data:
        method = getattr(supervised, method,
                         getattr(cluster_methods, method, None))
        if issubclass(method, cluster_methods.Method):
            context_vectors = get_context_vectors(
                word, os.path.join(ad_root, 'contexts-100k', word + '.txt'),
                weights)
            cluster_model = method(dict(
                word=word,
                ad_root=ad_root,
                context_vectors=context_vectors),
                n_senses=12)
            cluster_model.cluster()
            model = SupervisedWrapper(
                cluster_model, weights=weights, **model_params)
        else:
            model = method(train_data, weights=weights, **model_params)
    return model


def evaluate_word(word, ad_root, labeled_root,
                  print_errors=False, tsne=False, **model_params):
    senses, test_data = get_labeled_ctx(
        os.path.join(labeled_root, word + '.txt'))
    mfs_baseline = get_mfs_baseline(test_data)
    ad_word_data = get_ad_word(word, ad_root)
    if not ad_word_data: return
    model = train_model(word, ad_word_data, ad_root, **model_params)
    if not model: return
    test_accuracy, max_freq_error, js_div, estimate, answers = \
        evaluate(model, test_data)
    if tsne:
        show_tsne(model, answers, senses, word)
        # train_data = get_ad_train_data(word, ad_word_data)
        # show_tsne(model, [(x, ans, ans) for x, ans in train_data], senses, word)
    if print_errors:
        _print_errors(test_accuracy, answers, ad_word_data, senses)
    return mfs_baseline, model.get_train_accuracy(verbose=False), \
           test_accuracy, max_freq_error, js_div, estimate


def evaluate_words(filename, **params):
    with open(filename, 'r') as f:
        words = [l.strip() for l in f]
    all_metrics = []
    metric_names = ['MFS', 'Train', 'Test', 'Freq'] #, 'JSD', 'Estimate']
    wjust = 20
    print(u'\t'.join(['Word'.ljust(wjust)] + metric_names))
    for word in sorted(words):
        metrics = evaluate_word(word, **params)[:len(metric_names)]
        if metrics is not None:
            all_metrics.append(metrics)
            print(u'%s\t%s' % (
                word.ljust(wjust), '\t'.join('%.2f' % v for v in metrics)))
        else:
            print(u'%s\tmissing' % word)
    print(u'%s\t%s' % ('Avg.'.ljust(wjust), '\t'.join(
        '%.3f' % avg(metrics[i] for metrics in all_metrics)
        for i, _ in enumerate(metric_names))))


def run_on_words(ctx_dir, **params):
    for ctx_filename in os.listdir(ctx_dir):
        if not ctx_filename.endswith('.txt'):
            continue
        success = run_on_word(ctx_filename, ctx_dir, **params)
        if not success:
            print('skip', ctx_filename, file=sys.stderr)


def run_on_word(ctx_filename, ctx_dir, ad_root, **params):
    max_contexts = params.get('max_contexts')
    min_contexts = params.get('min_contexts')
    word = ctx_filename.split('.')[0]
    if word[-1].isdigit():
        return
    result_filename = os.path.join(ctx_dir, word + '.json')
    if os.path.exists(result_filename):
        print(result_filename, "already exists, skiping", file=sys.stderr)
        return True
    with open(os.path.join(ctx_dir, ctx_filename), 'r') as f:
        contexts = [line.split('\t') for line in f]
    if max_contexts and len(contexts) > max_contexts:
        contexts = random.sample(contexts, max_contexts)
    elif not contexts or (min_contexts and len(contexts) < min_contexts):
        return
    ad_word_data = get_ad_word(word, ad_root)
    if ad_word_data is None: return
    model = train_model(word, ad_word_data, ad_root, **params)
    if model is None: return
    result = []
    confidences = []
    for x in contexts:
        model_ans, confidence = model(x, with_confidence=True)
        result.append((x, model_ans))
        confidences.append(confidence)
    with open(result_filename, 'w') as f:
        json.dump({
            'word': word,
            'contexts': result,
            'estimate': get_accuracy_estimate(
                confidences, model.confidence_threshold),
            }, f)
    return True


def summary(ad_root, ctx_dir):
    all_freqs = {}
    word_ipm = load_ipm(ad_root)
    for filename in os.listdir(ctx_dir):
        if not filename.endswith('.json') or filename == 'summary.json':
            continue
        with open(os.path.join(ctx_dir, filename), 'r') as f:
            result = json.load(f)
            word = result['word']
            w_meta = get_ad_word(word, ad_root)
            meaning_by_id = {
                m['id']: m['meaning'] for m in w_meta['meanings']}
            counts = Counter(ans for _, ans in result['contexts'])
            all_freqs[word] = {
                'senses': {
                    ans: dict(
                        meaning=meaning_by_id[ans],
                        freq=cnt / len(result['contexts']))
                    for ans, cnt in counts.items()},
                'estimate': result.get('estimate'),
                'is_homonym': w_meta.get('is_homonym', False),
                'ipm': word_ipm.get(word, 0.0),
            }
    with open(os.path.join(ctx_dir, 'summary.json'), 'w') as f:
        json.dump(all_freqs, f)


def load_ipm(ad_root, only_pos='s'):
    filename = os.path.join(ad_root, 'freqs.csv')
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return {word: float(ipm)
                    for word, pos, ipm in csv.reader(f) if pos == only_pos}
    else:
        return {}


def _print_errors(test_accuracy, answers, ad_word_data, senses):
    errors = get_errors(answers)
    ad_senses = {m['id']: m['meaning'] for m in ad_word_data['meanings']}
    for sid, meaning in sorted_senses(senses):
        print()
        if sid in ad_senses:
            print(sid, ad_senses[sid])
        else:
            print(sid, meaning)
            print('Missing in AD!')
    assert set(ad_senses).issubset(senses)
    print('\ncorrect: %.2f\n' % test_accuracy)
    error_kinds = Counter((ans, model_ans) for _, ans, model_ans in errors)
    print('ans\tmodel\terrors')
    for (ans, model_ans), count in \
            sorted(error_kinds.items(), key=itemgetter(1), reverse=True):
        print('%s\t%s\t%s' % (ans, model_ans, count))
    print('\n')
    print('\t'.join(['ans', 'model_ans', 'ok?', 'left', 'word', 'right']))
    for (left, word, right), ans, model_ans in answers:
        print('\t'.join([ans, model_ans, 'error' if ans == model_ans else 'ok',
                         left, word, right]))
    print('\n')


def get_ad_train_data(word, ad_word_data):
    train_data = []
    for m in ad_word_data['meanings']:
        ans = m['id']
        for ctx in m['contexts']:
            before, mid, after = [], None, []
            append_to = before
            for item in mystem.analyze(ctx):
                text = item['text']
                if any(a['lex'] == word for a in item.get('analysis', [])):
                    mid = text
                    append_to = after
                elif text != '\n':
                    append_to.append(text)
            before, after = ''.join(before).strip(), ''.join(after).strip()
            train_data.append(((before, mid or word, after), ans))
    return train_data


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('action', choices=['run', 'summary', 'evaluate'])
    arg('ad_root')
    arg('word_or_filename')
    arg('--window', type=int, default=10)
    arg('--min-contexts', type=int, default=100,
        help='(for run) skip files with less contexts')
    arg('--max-contexts', type=int, default=2000,
        help='(for run) max number of contexts in sample')
    arg('--verbose', action='store_true')
    arg('--print-errors', action='store_true')
    arg('--tsne', action='store_true')
    arg('--labeled-root')
    arg('--no-weights', action='store_true')
    arg('--w2v-weights', action='store_true')
    arg('--no-lemm', action='store_true')
    arg('--method', default='SphericalModel')
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in [
        'ad_root', 'window', 'verbose', 'no_weights', 'w2v_weights', 'method']}
    params['lemmatize'] = not args.no_lemm
    if args.action == 'evaluate':
        params.update({k: getattr(args, k) for k in [
            'labeled_root', 'print_errors', 'tsne']})
        if not args.labeled_root:
            parser.error('Please specify --labeled-root')
        if os.path.exists(args.word_or_filename):
            evaluate_words(args.word_or_filename, **params)
        else:
            evaluate_word(args.word_or_filename, print_errors=True, **params)
    elif args.action == 'run':
        run_on_words(args.word_or_filename, **params)
    elif args.action == 'summary':
        summary(args.ad_root, args.word_or_filename)


if __name__ == '__main__':
    main()
