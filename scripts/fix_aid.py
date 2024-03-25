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

        question = line.pop('text')
        img_path = line.pop('image')

        line.update(question_id=idx,
                    question=question,
                    answer=gt,
                    image_path=img_path)

        fixed.append(line)

    dump(fixed, out_file)


def fix_ucmerced():
    inp_file = 'data/GeoChat-Bench/UCmerced.jsonl'
    out_file = 'data/GeoChat-Bench/UCmerced_fixed.jsonl'

    df = load(inp_file)

    fixed = []
    for idx, line in enumerate(df):
        gt = line.pop('ground_truth').capitalize()
        question = line.pop('text')
        img_path = line.pop('image')

        line.update(question_id=idx,
                    question=question,
                    answer=gt,
                    image_path=img_path)

        fixed.append(line)

    dump(fixed, out_file)


def main():
    fix_aid()
    fix_ucmerced()


if __name__ == '__main__':
    main()
