# encoding: utf-8

import random
import os.path

import xlrd


wb = xlrd.open_workbook(
    '/Users/kostia/Downloads/20-AD-RNC.xlsx.xlsx')

for sheet in wb.sheets():
    word = sheet.name
    def contexts():
        with open(os.path.join(
                'ad-nouns', 'RNC', word.encode('utf-8') + '.txt'), 'rb') as f:
            ctxs = list(f)
            random.seed(1)
            random.shuffle(ctxs)
            ctxs = ctxs[:100]
            for x in ctxs:
                yield x.decode('utf-8').strip()
    contexts = contexts()
   #contexts = iter(open(os.path.join(
   #    'ad-nouns', 'RNC', word.encode('utf-8') + '.txt'), 'rb'))
    is_header = True
    with open(os.path.join(
            'ann', 'RNC', word.encode('utf-8') + '.txt'), 'wb') as f:
        write = lambda lst: f.write(
            '\t'.join(map(unicode, lst)).encode('utf-8') + '\n')
        for row_n in xrange(sheet.nrows):
            row = [cell.value for cell in sheet.row(row_n)]
            if row[0] == '':
                is_header = False
                continue
            if is_header:
                sense_id, sense = row[:2]
                write(['', '', sense, int(sense_id)])
            else:
                ctx = next(contexts)
                sense, full_ctx = int(row[0]), row[2]
                is_valid = full_ctx.startswith(ctx.split('\t')[0].strip())
                if not is_valid:
                    print full_ctx
                    print ctx.split('\t')[0]
                    import pdb; pdb.set_trace()
                assert is_valid
                f.write(ctx.encode('utf-8') + '\t' + str(sense) + '\n')
