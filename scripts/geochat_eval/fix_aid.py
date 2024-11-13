import re
import sys

sys.path.append('.')

from piechat.utils import load, dump


def fix_aid():
    aid_file = 'data/GeoChat-Bench/aid.jsonl'
    out_file = 'data/GeoChat-Bench/aid_fixed.jsonl'

    df = load(aid_file)

    fixed = []
    for idx, line in enumerate(df):
        gt = line.pop('ground_truth')[:-2]
        gt = re.sub(r'([A-Z])', r' \1', gt).strip()

        line['ground_truth'] = gt
        fixed.append(line)

    dump(fixed, out_file)


def main():
    fix_aid()


if __name__ == '__main__':
    main()
