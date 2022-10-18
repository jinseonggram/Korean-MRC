import json
import copy


def parser(data: list):
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
    parsing_data_list = []
    for item in data:
        format_val = copy.deepcopy(sample_format)
        format_val['title'] = item['doc_title']
        format_val['news_category'] = item['doc_class']['code']
        format_val['source'] = item['doc_source']
        format_val['paragraphs'][0]['context'] = item['paragraphs'][0]['context']
        format_val['paragraphs'][0]['qas'][0]['question'] = item['paragraphs'][0]['qas'][0]['question']
        format_val['paragraphs'][0]['qas'][0]['guid'] = 'ai-hub-' + str(item['paragraphs'][0]['qas'][0]['question_id'])
        format_val['paragraphs'][0]['qas'][0]['answers'][0]['text'] = item['paragraphs'][0]['qas'][0]['answers']['text']
        format_val['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = int(item['paragraphs'][0]['qas'][0]['answers']['answer_start'])

        # answer 의 첫번째 element 값고 context[answer_start] 의 값이 같을 경우에만 저장하도록 조건
        if format_val['paragraphs'][0]['context'][format_val['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']] == list(format_val['paragraphs'][0]['qas'][0]['answers'][0]['text'])[0]:
            # answer text 의 첫번째 element 부분부터 answer text 의 끝 지점이 context 에 존재한다면 저장하는 조건
            source = ''
            context_list = list(format_val['paragraphs'][0]['context'])
            answer_text = format_val['paragraphs'][0]['qas'][0]['answers'][0]['text']
            start = format_val['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']
            for i in context_list[start:]:
                source += i
                if answer_text in source:
                    parsing_data_list.append(format_val)
                    break

    return parsing_data_list


# get file
with open('./TL_span_extraction.json', 'r') as f:
    extraction_row_file = json.load(f)

with open('./TL_unanswerable.json', 'r') as f:
    unanswerable_row_file = json.load(f)

# extract data list
extraction_data = extraction_row_file['data']  # 111967 개
unanswerable_data = unanswerable_row_file['data']  # 8000 개

# change data list to preprocessing for train format
extraction_list = parser(extraction_data)
unanswerable_list = parser(unanswerable_data)

# combine extraction_list with unanswerable_list
data_list = extraction_list + unanswerable_list
print(len(data_list))

# write file (train dataset + aihub mrc data set)
with open('./train.json', 'r') as f:
    train = json.load(f)

# train['data'] = data_list + train['data']
train['data'] = data_list + train['data']

file_path = "train_added_aihub.json"
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(train, file, ensure_ascii=False, indent='\t')
