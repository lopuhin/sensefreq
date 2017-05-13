#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import openpyxl


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input', help='.xlsx labeled contexts, one word on each sheet')
    arg('output', type=Path,
        help='output directory: one .json file for each word will be created')
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True)
    wb = openpyxl.load_workbook(args.input, read_only=True)
    for word in wb.get_sheet_names():
        ws = wb.get_sheet_by_name(word)
        cell = lambda row, col: ws.cell(row=row, column=col).value
        to_sense = lambda x: str(int(x))
        contexts_start = 1
        while cell(contexts_start, 1) is None:
            contexts_start += 1
            if contexts_start > 100:
                raise ValueError('Could not find contexts start for {}'.format(word))

        senses = {to_sense(cell(i, 3)): cell(i, 2) for i in range(1, contexts_start)}
        contexts = []
        row = contexts_start
        while cell(row, 1) is not None:
            contexts.append(
                ((cell(row, 2) or '', cell(row, 3) or '', cell(row, 4) or ''),
                 to_sense(cell(row, 1))))
            row += 1

        print('{}: {} senses, {} contexts'.format(word, len(senses), len(contexts)))
        args.output.joinpath('{}.json'.format(word)).write_text(
            json.dumps([senses, contexts], indent=True, ensure_ascii=False),
            encoding='utf8')



if __name__ == '__main__':
    main()
