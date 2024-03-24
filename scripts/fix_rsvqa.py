import json

import sys

sys.path.append('.')

from piechat.utils import load, dump


def load_jsonl(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]
    if lines[-1] == '':
        lines = lines[:-1]
    data = [json.loads(x) for x in lines]
    return data


def load_json(data_path):
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    return data


def main():
    answers_data_path = 'data/RSVQA/HR/USGS_split_test_phili_answers.json'
    images_data_path = 'data/RSVQA/HR/USGS_split_test_phili_images.json'
    questions_data_path = 'data/RSVQA/HR/USGS_split_test_phili_questions.json'

    answers = load(answers_data_path)['answers']
    images = load(images_data_path)['images']
    questions = load(questions_data_path)['questions']

    answers = [item for item in answers if item['active']]
    images = [item for item in images if item['active']]
    questions = [item for item in questions if item['active']]

    results = []
    # question: id, img_id, type, question, answers_ids
    # answer: question_id, answer
    for idx, question in enumerate(questions):
        question_id = question['id']
        img_id = question['img_id']
        question_type = question['type']
        question_text = question['question']
        answer = answers[idx]['answer']

        if question_type == 'area' or question_type == 'count':
            continue

        cur_item = dict(
            question_id=question_id,
            img_id=img_id,
            img_path=f'{img_id}.tif',
            category=question_type,
            question=
            f'{question_text}\nAnswer the question using a single word or phrase.',
            ground_truth=answer,
        )

        results.append(cur_item)

    dump(results, 'data/GeoChat-Bench/hrben_fixed.jsonl')


if __name__ == '__main__':
    main()
