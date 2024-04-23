import re
import sys

sys.path.append('.')

from piechat.utils import load, dump


def fix_aid():
    aid_file = 'data/GeoChat-Bench/aid.jsonl'
    out_file = 'data/PIEGeo_Bench/aid.jsonl'

    df = load(aid_file)

    fixed = []
    for idx, line in enumerate(df):
        gt = line.pop('ground_truth')[:-2]
        gt = re.sub(r'([A-Z])', r' \1', gt).strip()

        question = line.pop('text')
        img_path = line.pop('image')

        new_line = dict(index=idx, image=img_path, answer=gt)
        fixed.append(new_line)

    dump(fixed, out_file)


def fix_ucmerced():
    inp_file = 'data/GeoChat-Bench/UCmerced.jsonl'
    out_file = 'data/PIEGeo_Bench/UCmerced.jsonl'

    df = load(inp_file)

    fixed = []
    for idx, line in enumerate(df):
        gt = line.pop('ground_truth').capitalize()
        question = line.pop('text')
        img_path = line.pop('image')

        new_line = dict(index=idx, image=img_path, answer=gt)
        fixed.append(new_line)

    dump(fixed, out_file)


def main():
    fix_aid()
    fix_ucmerced()


if __name__ == '__main__':
    main()
