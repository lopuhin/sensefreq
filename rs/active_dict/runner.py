#!/usr/bin/env python
from __future__ import print_function
import argparse
from collections import Counter, defaultdict
import csv
import json
from operator import itemgetter
import os.path
from pathlib import Path
import random
import re
import sys

from rs.utils import avg
from rlwsd.utils import mystem
from rs.active_dict.loader import get_ad_word
from rs.supervised import get_labeled_ctx, evaluate, load_weights, get_errors, \
    SupervisedWrapper, sorted_senses, get_accuracy_estimate, get_mfs_baseline, \
    show_tsne
from rs.cluster import get_context_vectors
from rs import cluster_methods, supervised


def train_model(word, train_data, ad_root, method=None, **model_params):
    weights = None if model_params.pop('no_weights', None) else load_weights(
        word, root=ad_root, lemmatize=model_params.get('lemmatize'))
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
                  print_errors=False, tsne=False, coarse=False,
                  alt_root=None, alt_senses=False, **model_params):
    word_path = labeled_root.joinpath(word + '.json')
    if not word_path.exists():
        word_path = labeled_root.joinpath(word + '.txt')
    senses, test_data = get_labeled_ctx(str(word_path))
    mfs_baseline = get_mfs_baseline(test_data)
    ad_word_data = get_ad_word(word, ad_root)
    if not ad_word_data:
        print(word, 'no AD data', sep='\t')
        return
    ad_senses = {str(i): m['name']
                 for i, m in enumerate(ad_word_data['meanings'], 1)}
    if set(ad_senses) != set(senses):
        print(word, 'AD/labeled sense mismatch', sep='\t')
        return
    train_data = get_ad_train_data(word, ad_word_data)
    if alt_root:
        senses, test_data, train_data = get_alt_senses_test_train_data(
            alt_root, word, senses, test_data, train_data, alt_senses=alt_senses)
    if coarse:
        sense_mapping = get_coarse_sense_mapping(ad_senses)
        inverse_mapping = defaultdict(list)
        for old_id, new_id in sense_mapping.items():
            inverse_mapping[new_id].append(old_id)
        senses = {new_id: '; '.join(senses[old_id] for old_id in old_ids)
                  for new_id, old_ids in inverse_mapping.items()}
        train_data = [(ctx, sense_mapping[old_id]) for ctx, old_id in train_data]
        test_data = [(ctx, sense_mapping[old_id]) for ctx, old_id in test_data]
    model = train_model(word, train_data, ad_root, **model_params)
    if not model:
        print(word, 'no model', sep='\t')
        return
    test_accuracy, max_freq_error, js_div, estimate, answers = \
        evaluate(model, test_data)
    if tsne:
        show_tsne(model, answers, senses, word)
        # train_data = get_ad_train_data(word, ad_word_data)
        # show_tsne(model, [(x, ans, ans) for x, ans in train_data], senses, word)
    if print_errors:
        _print_errors(test_accuracy, answers, ad_word_data, senses)
    return (len(senses), mfs_baseline, model.get_train_accuracy(verbose=False),
            test_accuracy, max_freq_error, js_div, estimate)


def get_coarse_sense_mapping(senses):
    mapping = defaultdict(list)
    for sense_id, sense_name in sorted(senses.items()):
        key = tuple(re.findall(r'\d+(?:\.\d+)?\b', sense_name.split(':', 1)[0]))
        assert key, sense_name
        coarse_key = tuple(s.split('.', 1)[0] for s in key)
        mapping[coarse_key].append(sense_id)
    return {old_sense_id: str(new_sense_id)
            for new_sense_id, (_, old_sense_ids)
            in enumerate(sorted(mapping.items()), 1)
            for old_sense_id in old_sense_ids}


def evaluate_words(words, **params):
    all_metrics = []
    metric_names = ['senses', 'MFS', 'Train', 'Test', 'Freq'] #, 'JSD', 'Estimate']
    wjust = 20
    print(u'\t'.join(['Word'.ljust(wjust)] + metric_names))
    for word in sorted(words):
        metrics = evaluate_word(word, **params)
        if metrics is not None:
            metrics = metrics[:len(metric_names)]
            all_metrics.append(metrics)
            print(u'%s\t%s' % (
                word.ljust(wjust), '\t'.join(
                    ('%d' if isinstance(v, int) else '%.2f') % v for v in metrics)))
        else:
            pass  # print(u'%s\tmissing' % word)
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
    if ad_word_data is None:
        return
    train_data = get_ad_train_data(word, ad_word_data)
    model = train_model(word, train_data, ad_root, **params)
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
            }, f, ensure_ascii=False)
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
        json.dump(all_freqs, f, ensure_ascii=False)


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
            train_data.append((_get_training_ctx(ctx, word), ans))
    return train_data


def _get_training_ctx(ctx, word):
    before, mid, after = [], None, []
    append_to = before
    for item in mystem.analyze(ctx):
        text = item['text']
        if any(a['lex'] == word for a in item.get('analysis', [])):
            mid = text
            append_to = after
        elif text != '\n':
            append_to.append(text)
    before, after = [''.join(x).strip().replace('\xad', '')
                     for x in [before, after]]
    return before, mid or word, after


def get_alt_senses_test_train_data(alt_root, word, senses, test_data,
                                   train_data, alt_senses=False):
    """ Load alternative dictionary train data.
    Example format:
    1
    ex1
    ex2

    2,3
    ex1

    Returns senses, test_data, train_data.
    """
    alt_def = alt_root.joinpath('{}.txt'.format(word)).read_text(encoding='utf8')
    new_train_data = []
    sense_mapping = {}
    new_senses = {}
    for sense_id, alt_sense in enumerate(alt_def.split('\n\n'), 1):
        sense_id = str(sense_id)
        ad_ids, *examples = alt_sense.split('\n')
        for ad_id in ad_ids.split(','):
            assert int(ad_id) > 0
            assert ad_id not in sense_mapping
            sense_mapping[ad_id] = sense_id
        new_train_data.extend(filter(None, (
            (_get_training_ctx(ctx, word), sense_id) for ctx in examples)))
        new_senses[sense_id] = 'AD: {}'.format(ad_ids)
    test_data = [(ctx, sense_mapping[ad_id]) for ctx, ad_id in test_data
                 if ad_id in sense_mapping]
    if alt_senses:
        new_train_data = [(ctx, sense_mapping[ad_id]) for ctx, ad_id in train_data
                          if ad_id in sense_mapping]
    return new_senses, test_data, new_train_data


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('action', choices=['run', 'summary', 'evaluate'])
    arg('ad_root')
    arg('word_or_filename', nargs='?')
    arg('--window', type=int, default=10)
    arg('--min-contexts', type=int, default=100,
        help='(for run) skip files with less contexts')
    arg('--max-contexts', type=int, default=2000,
        help='(for run) max number of contexts in sample')
    arg('--verbose', action='store_true')
    arg('--print-errors', action='store_true')
    arg('--tsne', action='store_true')
    arg('--labeled-root', type=Path)
    arg('--no-weights', action='store_true')
    arg('--w2v-weights', action='store_true')
    arg('--no-lemm', action='store_true')
    arg('--method', default='SphericalModel')
    arg('--coarse', action='store_true', help='merge fine-grained senses')
    arg('--alt-root', type=Path, help='alternative dictionary root')
    arg('--alt-senses', action='store_true', help='use alternative senses for AD')
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in [
        'ad_root', 'window', 'verbose', 'no_weights', 'w2v_weights', 'method']}
    params['lemmatize'] = not args.no_lemm
    if args.action == 'evaluate':
        params.update({k: getattr(args, k) for k in [
            'labeled_root', 'print_errors', 'tsne', 'coarse', 'alt_root', 'alt_senses']})
        if not args.labeled_root:
            parser.error('Please specify --labeled-root')
        if not args.word_or_filename:
            words = [p.stem for pattern in ['*.txt', '*.json']
                     for p in args.labeled_root.glob(pattern)]
        elif os.path.exists(args.word_or_filename):
            with open(args.word_or_filename, 'rt') as f:
                words = [l.strip() for l in f]
        else:
            words = [args.word_or_filename]
        evaluate_words(words, **params)
    else:
        if args.coarse:
            parser.error('--coarse supported only for "evaluate" action')
        if not args.word_or_filename:
            parser.error('Please specify word_or_filename')
        if args.action == 'run':
            run_on_words(args.word_or_filename, **params)
        elif args.action == 'summary':
            summary(args.ad_root, args.word_or_filename)


if __name__ == '__main__':
    main()
