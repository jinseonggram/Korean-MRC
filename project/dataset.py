import json
import random
from typing import List, Tuple, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils.s3 import load_json_from_s3

tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')


class KoMRC:
    """
    Data load & split
    """
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        # with open(file_path, 'r', encoding='utf-8') as fd:
        #     data = json.load(fd)
        data = load_json_from_s3(file_path)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))

        return cls(data, indices)

    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float = .1):
        indices = list(dataset._indices)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio):]
        eval_indices = indices[:int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        qa = paragraph['qas'][q_id]

        guid = qa['guid']

        context = paragraph['context'].replace('\n', 'n').replace('\xad', '')
        question = qa['question'].replace('\n', 'n').replace('\xad', '')
        answers = qa['answers']
        if answers != None:
            for a in answers:
                a['text'] = a['text'].replace('\n', 'n').replace('\xad', '')


        return {'guid': guid,
                'context': context,
                'question': question,
                'answers': answers
                }

    def __len__(self) -> int:
        return len(self._indices)


class TokenizedKoMRC(KoMRC):
    """
    Tokenizer & Tag Token Position
    """
    def __init__(self, data, indices: List[Tuple[int, int, int]]) -> None:
        super().__init__(data, indices)
        self._tokenizer = tokenizer

    def _tokenize_with_position(self, sentence: str) -> List[Tuple[str, Tuple[int, int]]]:
        position = 0
        tokens = []

        sentence_tokens = []
        for word in sentence.split():
            if '[UNK]' in self._tokenizer.tokenize(word):
                sentence_tokens.append(word)
            else:
                sentence_tokens += self._tokenizer.tokenize(word)

        for morph in sentence_tokens:
            if len(morph) > 2:
                if morph[:2] == '##':
                    morph = morph[2:]

            position = sentence.find(morph, position)
            tokens.append((morph, (position, position + len(morph))))
            position += len(morph)

        return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        # sample = {'guid': guid, 'context': context, 'question': question, 'answers': answers}

        context, position = zip(*self._tokenize_with_position(sample['context']))
        context, position = list(context), list(position)

        question = self._tokenizer.tokenize(sample['question'])

        if sample['answers'] is not None:
            answers = []
            for answer in sample['answers']:
                for start, (position_start, position_end) in enumerate(position):
                    if position_start <= answer['answer_start'] < position_end:
                        break
                else:
                    raise ValueError("No mathced start position")

                target = ''.join(answer['text'].split(' '))
                source = ''
                for end, morph in enumerate(context[start:], start):
                    source += morph
                    if target in source:
                        break
                else:
                    raise ValueError("No Matched end position")

                answers.append({'start': start, 'end': end})

        else:
            answers = None

        return {
            'guid': sample['guid'],
            'context_original': sample['context'],
            'context_position': position,
            'question_original': sample['question'],
            'context': context,
            'question': question,
            'answers': answers
        }


class Indexer:
    """
    Input
    """
    def __init__(self, vocabs: List[str], max_length: int):
        self.max_length = max_length
        self.vocabs = vocabs
        self._tokenizer = tokenizer

    @property
    def vocab_size(self):
        return len(self.vocabs)

    @property
    def pad_id(self):
        return self._tokenizer.vocab['[PAD]']

    @property
    def unk_id(self):
        return self._tokenizer.vocab['[UNK]']

    @property
    def cls_id(self):
        return self._tokenizer.vocab['[CLS]']

    @property
    def sep_id(self):
        return self._tokenizer.vocab['[SEP]']

    def sample2ids(self, sample: Dict[str, Any], ) -> Dict[str, Any]:
        context = [self._tokenizer.convert_tokens_to_ids(token) for token in sample['context']]
        question = [self._tokenizer.convert_tokens_to_ids(token) for token in sample['question']]

        context = context[:self.max_length - len(question) - 3]  # Truncate context

        input_ids = [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)

        if sample['answers'] is not None:
            answer = sample['answers'][0]
            start = min(len(question) + 2 + answer['start'], self.max_length - 1)
            end = min(len(question) + 2 + answer['end'], self.max_length - 1)
        else:
            start = None
            end = None

        return {
            'guid': sample['guid'],
            'context': sample['context_original'],
            'question': sample['question_original'],
            'position': sample['context_position'],
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'start': start,
            'end': end
        }


class IndexerWrappedDataset:
    def __init__(self, dataset: TokenizedKoMRC, indexer: Indexer) -> None:
        self._dataset = dataset
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._indexer.sample2ids(self._dataset[index])
        sample['attention_mask'] = [1] * len(sample['input_ids'])

        return sample


class Collator:
    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = {key: [sample[key] for sample in samples] for key in samples[0]}

        for key in 'start', 'end':
            if samples[key][0] is None:
                samples[key] = None
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)

        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            samples[key] = pad_sequence([torch.tensor(sample, dtype=torch.long) for sample in samples[key]],
                                        batch_first=True,
                                        padding_value=self._indexer.pad_id)

        return samples