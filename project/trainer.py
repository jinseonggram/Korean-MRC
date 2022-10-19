import csv
import random
from statistics import mean
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.initialize_seed import seed_everything
from utils.s3 import write_json_to_s3

from model import get_model, get_tokenizer
from dataset import TokenizedKoMRC, IndexerWrappedDataset, Indexer, Collator
import wandb


def train(parameters: dict):
    epochs = parameters['epochs']
    train_losses = []
    dev_losses = []

    train_loss = []
    dev_loss = []

    loss_accumulate = 0.
    best_model = [-1, int(1e9)]
    global_step = 0

    # tokenize
    tokenizer = get_tokenizer(parameters['model'])

    # dataloader
    train_dataset_filename = parameters['datasets'][0]
    dataset = TokenizedKoMRC.load(train_dataset_filename)
    train_dataset, dev_dataset = TokenizedKoMRC.split(dataset)
    indexer = Indexer(list(tokenizer.vocab.keys()), parameters['max_length'])
    indexed_train_dataset = IndexerWrappedDataset(train_dataset, indexer)
    indexed_dev_dataset = IndexerWrappedDataset(dev_dataset, indexer)
    collator = Collator(indexer)
    train_loader = DataLoader(indexed_train_dataset,
                              batch_size=parameters['batch_size']['train'] // parameters['accumulate'],
                              shuffle=True,
                              collate_fn=collator,
                              num_workers=2)
    dev_loader = DataLoader(indexed_dev_dataset,
                            batch_size=parameters['batch_size']['eval'],
                            shuffle=False,
                            collate_fn=collator,
                            num_workers=2)

    # model
    model = get_model(parameters['model'])
    model.cuda()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'])

    # seed initialize
    seed_everything(parameters['seed'])

    for epoch in range(epochs):
        print("Epoch", epoch, '===============================================================================================================')

        # Train
        progress_bar_train = tqdm(train_loader, desc='Train')
        for i, batch in enumerate(progress_bar_train, 1):
            del batch['guid'], batch['context'], batch['question'], batch['position']
            batch = {key: value.cuda() for key, value in batch.items()}

            start = batch.pop('start')
            end = batch.pop('end')

            output = model(**batch)

            start_logits = output.start_logits
            end_logits = output.end_logits

            loss = (F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)) / parameters['accumulate']
            loss.backward()

            loss_accumulate += loss.item()

            global_step += 1

            del batch, start, end, start_logits, end_logits, loss

            if i % parameters['accumulate'] == 0:
                # clip_grad_norm_(model.parameters(), max_norm=1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)

                train_loss.append(loss_accumulate)
                progress_bar_train.set_description(f"Train - Loss: {loss_accumulate:.3f}, global: {global_step}")
                loss_accumulate = 0.
            else:
                continue

            if i % int(len(train_loader) / (parameters['accumulate'] * 25)) == 0:
                # Evaluation
                for batch in dev_loader:
                    del batch['guid'], batch['context'], batch['question'], batch['position']
                    batch = {key: value.cuda() for key, value in batch.items()}

                    start = batch.pop('start')
                    end = batch.pop('end')

                    model.eval()
                    with torch.no_grad():
                        output = model(**batch)

                        start_logits = output.start_logits
                        end_logits = output.end_logits
                    model.train()

                    loss = F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)

                    dev_loss.append(loss.item())

                    del batch, start, end, start_logits, end_logits, loss

                train_losses.append(mean(train_loss))
                dev_losses.append(mean(dev_loss))
                train_loss = []
                dev_loss = []

                if dev_losses[-1] <= best_model[1]:
                    best_model = (epoch, dev_losses[-1])
                    model.save_pretrained(f'models/{parameters["NAME"]}_{epoch}')

                    # Save
                    torch.save(model.state_dict(), "kobigbird-model.pth")

                # wandb
                wandb.log({"train_loss": train_losses[-1], "valid_loss": dev_losses[-1]})

        wandb.finish()
        print(f"Train Loss: {train_losses[-1]:.3f}")
        print(f"Valid Loss: {dev_losses[-1]:.3f}")
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    return best_model, train_losses, dev_losses


def test(parameter, best_model):
    start_visualize = []
    end_visualize = []

    # tokenize
    tokenizer = get_tokenizer(parameters['model'])

    # dataset
    test_dataset_filename = parameters['datasets'][1]
    test_dataset = TokenizedKoMRC.load(test_dataset_filename)
    indexer_test = Indexer(list(tokenizer.vocab.keys()), parameters['max_length'])
    indexed_test_dataset = IndexerWrappedDataset(test_dataset, indexer_test)

    print("Number of Test Samples", len(test_dataset))
    # print(test_dataset[0])

    # model
    model = get_model(f'models/{parameter["NAME"]}_{best_model[0]}')
    model.cuda()

    rows = []

    for sample in tqdm(indexed_test_dataset, "Testing"):
        input_ids, token_type_ids = [torch.tensor(sample[key], dtype=torch.long, device="cuda") for key in
                                     ("input_ids", "token_type_ids")]

        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :])

        start_logits = output.start_logits
        end_logits = output.end_logits
        start_logits.squeeze_(0), end_logits.squeeze_(0)

        start_prob = start_logits[token_type_ids.bool()][1:-1].softmax(-1)
        end_prob = end_logits[token_type_ids.bool()][1:-1].softmax(-1)

        probability = torch.triu(start_prob[:, None] @ end_prob[None, :])

        # 토큰 길이 8까지만
        for row in range(len(start_prob) - 8):
            probability[row] = torch.cat(
                (probability[row][:8 + row].cpu(), torch.Tensor([0] * (len(start_prob) - (8 + row))).cpu()), 0)

        index = torch.argmax(probability).item()

        start = index // len(end_prob)
        end = index % len(end_prob)

        # 확률 너무 낮으면 자르기
        if start_prob[start] > 0.3 and end_prob[end] > 0.3:
            start_str = sample['position'][start][0]
            end_str = sample['position'][end][1]
        else:
            start_str = 0
            end_str = 0

        start_visualize.append((list(start_prob.cpu()), (start, end), (start_str, end_str)))
        end_visualize.append((list(end_prob.cpu()), (start, end), (start_str, end_str)))

        rows.append([sample["guid"], sample['context'][start_str:end_str]])

    return rows


if __name__ == "__main__":
    # wandb
    wandb.login()

    # Train Parameter
    parameters = {
        "datasets": ["train_added_aihub.json", "test.json"],                            # 학습용 데이터 경로
        "epochs": 10,                # 전체 학습 Epoch
        'batch_size': {'train': 256,
                        'eval': 16,
                        'test': 256},         # Batch Size
        "learning_rate": 6e-5,
        'accumulate': 64,
        'seed': 42,
        'model': 'monologg/kobigbird-bert-base',
        'max_length': 4096,
        "train_output": "output"    # 학습된 모델의 저장 경로
    }
    parameters['NAME'] = f'kobigbird_ep{parameters["epochs"]}_max{parameters["max_length"]}_lr{parameters["learning_rate"]}_{random.randrange(0, 1024)}'
    parameters['save_path'] = f'./{parameters["NAME"]}'

    # wandb
    wandb.init()
    wandb.run.name = parameters['NAME']
    wandb.log({"train_loss": 0, "valid_loss": 0})

    # Train
    best_model, train_losses, dev_losses = train(parameters)

    # Test
    prediction_set = test(parameters, best_model)

    # Save
    data = {
        'train_losses': train_losses,
        'dev_losses': dev_losses,
        'prediction_set': prediction_set
    }
    write_json_to_s3(f'{parameters["NAME"]}', data)
    print("Train Complete")