
import argparse
from pathlib import Path
import logging
import json
import os
import pickle

from collections import defaultdict

from util import setup_logger
from search import main_search

TEMP_VOLUME_PATH = Path(__file__).resolve().parent.parent / Path("temp_volume")
INTERMEDIATE_OUTPUT_PATH = TEMP_VOLUME_PATH / "intermediate_output"

def append_sentences_to_csv(conllu_path, csv_path, sentence_indexes, overwrite=False):
    sentence_counter = 0
    current_block = []
    output_blocks = []

    with open(conllu_path, 'r', encoding='utf-8') as infile:
        next(infile, None)  # Skip the first header line
        for line in infile:
            if line.strip() == "":
                if current_block:
                    sentence_counter += 1
                    if sentence_counter in sentence_indexes:
                        output_blocks.append(current_block.copy())
                    current_block.clear()
            else:
                current_block.append(line.rstrip('\n'))
        if current_block:
            sentence_counter += 1
            if sentence_counter in sentence_indexes:
                output_blocks.append(current_block)

    mode = 'w' if overwrite or not os.path.exists(csv_path) else 'a'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, mode, encoding='utf-8', newline='') as outfile:
        for block in output_blocks:
            for line in block:
                outfile.write(line + '\n')
            outfile.write('\n')  # separate blocks


def main(args: argparse.Namespace) -> None:
    out = main_search(args)
    if args.print == 'json':
        print(json.dumps(out))
    elif args.print == 'kwic' and out['kwic']:
        if args.output_conllu_file:
            search_id_suffix = f"_{args.id}" if args.id is not None else ""

            INTERMEDIATE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

            sentence_idx = defaultdict(set)
            for match in out['kwic']:
                sentence_idx[match['corpus']].add(match['sentence'])

            for corpus in sentence_idx:
                output_filepath = INTERMEDIATE_OUTPUT_PATH / (corpus + f'_output{search_id_suffix}')
                output_indices_filepath = INTERMEDIATE_OUTPUT_PATH / (corpus + f'_output_indices{search_id_suffix}.pkl')
                if os.path.exists(output_filepath):
                    os.remove(output_filepath)
                if os.path.exists(output_indices_filepath):
                    os.remove(output_indices_filepath)

            for corpus, sentence in sentence_idx.items():
                append_sentences_to_csv(
                    Path(args.base_dir) / Path(corpus).with_suffix(".csv"),
                    INTERMEDIATE_OUTPUT_PATH / Path(corpus + '_output' + search_id_suffix),
                    sentence,
                    args.overwrite_file
                )

            if args.output_indices:
                for corpus, sentence in sentence_idx.items():
                    output_indices_filepath = INTERMEDIATE_OUTPUT_PATH / Path(corpus + '_output_indices' + search_id_suffix + '.pkl')
                    with open(output_indices_filepath, 'wb') as f:
                        # korpsearch sentence numbers are 1-based; deptreepy expects 0-based
                        pickle.dump(sorted(s - 1 for s in sentence), f)
        else:
            print(f"{out['hits']} search results. Showing n:o {out['start']}-{out['end']}:")
            for n, result in enumerate(out['kwic'], out['start']):
                match = result['match']
                print(f"{n}. [{result['sentence']}:{match['position']}]", end="")
                for i, token in enumerate(result['tokens']):
                    if i == match['start']:
                        print(" {", end="")
                    print(" " + "/".join(token.values()), end="")
                    if i + 1 == match['end']:
                        print(" }", end="")
                print()
    else:
        print(f"{out['hits']} search results.")


################################################################################
## Command-line arguments

parser = argparse.ArgumentParser(description='Test things')
parser.add_argument('--corpus', '-c', nargs='+', required=True,
    help='name(s) of compiled corpora to search in')
parser.add_argument('--query', '-q', type=str, required=True,
    help='the query (e.g., \'[pos="ART"] [lemma="small"] [pos="SUBST"]\')')

parser.add_argument('--base-dir', '-d', type=Path, metavar='DIR', default=Path('corpora'),
    help='directory where to find the corpus (default: ./corpora/)')
parser.add_argument('--cache-dir', type=Path, metavar='DIR', default=Path('cache'),
    help='directory where to store cache files (default: ./cache/)')

parser.add_argument('--print', '-p', choices=['kwic','json'], default='kwic',
    help='output format for search results (default: kwic = keywords in context)')
parser.add_argument('--start', '-s', type=int, default=0,
    help='index of first result (default: 0)')
parser.add_argument('--num', '-n', type=int, default=10,
    help='n:o of shown results (default: 10)')
parser.add_argument('--end', '-e', type=int,
    help='index of last result (default: decided by --start and --num)')
parser.add_argument('--show', '-f', type=str,
    help='comma-separated list of features to show (default: the ones in the query)')

parser.add_argument('--no-cache', action="store_true", help="don't use cached queries")
parser.add_argument('--no-binary', action="store_true", help="don't use binary indexes")
parser.add_argument('--verbose', '-v', action="store_const", dest="loglevel", const=logging.INFO,
    help='verbose output (default: quiet)')
parser.add_argument('--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING,
    help='debugging output (defaul: quiet)')

parser.add_argument('--no-sentence-breaks', action='store_true',
    help="don't care about sentence breaks (default: do care)")
parser.add_argument('--filter', action='store_true',
    help='filter the final results (should not be necessary, and can take time)')

parser.add_argument('--output-conllu-file', action='store_true',
    help='write matched sentences to a CoNLL-U file instead of printing KWIC results')
parser.add_argument('--overwrite-file', action='store_true',
    help='overwrite the output file instead of appending to it')
parser.add_argument('--output-indices', action='store_true',
    help='also write a pickle file of matched (0-based) sentence indices per corpus')
parser.add_argument('--id', type=str, default=None,
    help='optional search id, appended to output filenames to avoid clashes between concurrent searches')

if __name__ == '__main__':
    args = parser.parse_args()
    setup_logger('{relativeCreated:8.2f} s {warningname}| {message}', timedivider=1000, loglevel=args.loglevel)
    main(args)
