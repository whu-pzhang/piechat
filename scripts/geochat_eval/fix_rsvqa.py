import json

import sys

sys.path.append('.')

from piechat.utils import load, dump


def fix_rsvqa_hr():
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
            image=f'{img_id}.tif',
            text=
            f'{question_text}\nAnswer the question using a single word or phrase.',
            category=question_type,
            ground_truth=answer,
        )
        results.append(cur_item)

    dump(results, 'data/GeoChat-Bench/hrben_fixed.jsonl')


def fix_rsvqa_lr():
    answers_data_path = 'data/RSVQA/LR/LR_split_test_answers.json'
    images_data_path = 'data/RSVQA/LR/LR_split_test_images.json'
    questions_data_path = 'data/RSVQA/LR/LR_split_test_questions.json'

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

        if question_type == 'area':
            continue

        cur_item = dict(
            question_id=question_id,
            image=f'{img_id}.tif',
            text=
            f'{question_text}\nAnswer the question using a single word or phrase.',
            category=question_type,
            ground_truth=answer,
        )
        results.append(cur_item)

    dump(results, 'data/GeoChat-Bench/lrben_fixed.jsonl')


def main():
    fix_rsvqa_lr()
    fix_rsvqa_hr()


if __name__ == '__main__':
    main()
