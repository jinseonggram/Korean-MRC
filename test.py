import json
import copy

# with open('./TL_unanswerable.json', 'r') as f:
#     row_file = json.load(f)


def parser(data):
    sample_format = {
        'title': '',
        'paragraphs': [{
            'context': '',
            'qas': [
                {
                    'question': '',
                    'answers': [
                        {
                            'text': '',
                            'answer_start': 0  # int
                        }
                    ],
                    'guid': ''
                }
            ]
        }],
        "news_category": '',
        "source": ''
    }

    question_format = {
        'question': '',
        'answers': [
            {
                'text': '',
                'answer_start': 0  # int
            }
        ],
        'guid': ''
    }
    parsing_data_list = []
    questions_list = []

    for item in data:
        format_val = copy.deepcopy(sample_format)
        format_val['title'] = item['doc_title']
        format_val['news_category'] = item['doc_class']['code']
        format_val['source'] = item['doc_source']
        format_val['paragraphs'][0]['context'] = item['paragraphs'][0]['context'].replace('\n', '').replace('\"', '')

        # for idx, question in enumerate(item['paragraphs'][0]['qas']):
        #     question_format_dict = copy.deepcopy(question_format)
        #     question_format_dict['question'] = item['paragraphs'][0]['qas'][idx]['question']
        #     question_format_dict['guid'] = 'ai-hub-' + str(item['paragraphs'][0]['qas'][idx]['question_id'])
        #     question_format_dict['answers'][0]['text'] = item['paragraphs'][0]['qas'][idx]['answers']['text']
        #     question_format_dict['answers'][0]['answer_start'] = int(item['paragraphs'][0]['qas'][idx]['answers']['answer_start'])
        #     questions_list.append(question_format_dict)
        #
        # format_val['paragraphs'][0]['qas'] = questions_list[:]
        # questions_list.clear()

        format_val['paragraphs'][0]['qas'][0]['question'] = item['paragraphs'][0]['qas'][0]['question']
        format_val['paragraphs'][0]['qas'][0]['guid'] = 'ai-hub-' + str(item['paragraphs'][0]['qas'][0]['question_id'])
        format_val['paragraphs'][0]['qas'][0]['answers'][0]['text'] = item['paragraphs'][0]['qas'][0]['answers']['text']
        format_val['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = int(item['paragraphs'][0]['qas'][0]['answers']['answer_start'])
        parsing_data_list.append(format_val)

    return parsing_data_list


with open('./TL_span_extraction.json', 'r') as f:
    row_file = json.load(f)

data = row_file['data']  # 111967 개
extraction_list = parser(data)
print(len(extraction_list))

with open('./TL_unanswerable.json', 'r') as f:
    row_file = json.load(f)

data = row_file['data']  # 111967 개
unanswerable_list = parser(data)
print(len(unanswerable_list))

data_list = extraction_list + unanswerable_list
print(len(data_list))

with open('./train.json', 'r') as f:
    train = json.load(f)

train['data'] = data_list + train['data']
print(len(train['data']))
file_path = "train_added_aihub.json"

with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(train, file, ensure_ascii=False, indent='\t')