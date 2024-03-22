import jsonlines
import re

aid_file = 'data/GeoChat-Bench/aid.jsonl'

fixed = []

prev_gt = ''
with jsonlines.open(aid_file, 'r') as fp:
    for line in fp:
        gt = line['ground_truth'][:-2]
        gt = re.sub(r'([A-Z])', r' \1', gt).strip()

        line.update(ground_truth=gt)

        fixed.append(line)

out_file = 'data/GeoChat-Bench/aid_fixed.jsonl'

with jsonlines.open(out_file, 'w') as fp:
    fp.write_all(fixed)
