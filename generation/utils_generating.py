import torch
import os
import json
from tqdm import tqdm
from transformers import BertTokenizer
import logging
from itertools import chain
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>",
                         "<knowledge_tag>"]


class ResonseDataset(torch.utils.data.Dataset):
    def __init__(self,args,tokenizer):
        super(ResonseDataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.example,self.original_example = self.load_data()

    def load_data(self):

        # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        cached_train_feature_file = os.path.join(self.args.data_dir, self.args.task_name, f"cached_train_{self.args.max_len}")

        with open(os.path.join(self.args.data_dir, self.args.task_name, "train_data.json"), encoding="utf-8") as f:
            # train_data = json.load(f)
            train_data = json.load(f)[:200]

        if os.path.exists(cached_train_feature_file):
            logger.info("已经存在train缓存文件{}，直接加载".format(cached_train_feature_file))
            train_data_set = torch.load(cached_train_feature_file)["data_set"]
        # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info("不存在train缓存文件{}，进行数据预处理操作".format(cached_train_feature_file))

            train_data_set = self.process_data(train_data)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_train_feature_file))
            # torch.save({"data_set": train_data_set}, cached_train_feature_file)
        return train_data_set, train_data

    def process_data(self,data):
        data_set = list()
        for idx, sample in enumerate(tqdm(data, desc="iter", disable=False)):
            history, response, knowledge, dialog_id = sample["history"], sample["response"], sample["knowledge"], \
                                                      sample["dialog_id"]
            history_convert, index_cnt = list(), 0
            for i in range(len(history)):
                # 从后往前添加历史对话，如果添加对话超出最大长度限制了就切断，只尽可能的保留历史对话（两个人的）
                history_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(history[len(history) - i - 1]["text"])
                )
                index_cnt += len(history_ids)
                if index_cnt >= self.args.history_max_tokens:
                    red = self.args.history_max_tokens - (index_cnt - len(history_ids))
                    history_convert.append(history_ids[0:red])
                    break
                history_convert.append(history_ids)
            truncated_history = truncate_sequences(history_convert, self.args.history_max_tokens)
            # 回复内容
            response_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response))
            response_ids = response_ids[:self.args.resp_max_tokens]
            # 把用到的三元组拼起来，三元组内的元素用“-”连接，三元组之间用“；”连接
            # 也只保留最大长度个三元组
            if len(knowledge) == 0:
                used_knowledge = list()
            else:
                str_knowledge = ""
                for item in knowledge:
                    temp = "-".join([str(item["attrname"]), str(item["attrvalue"]), str(item["name"])])
                    str_knowledge = ";".join([str_knowledge, temp])
                used_knowledge = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(str_knowledge)
                )
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            data_set.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "response": response_ids,
                "response_text": response,
                "dialog_id": dialog_id
            })
        return data_set
        # with open(os.path.join(args.data_dir, args.task_name, "eval_data.json"), encoding="utf-8") as f:
        #     eval_data = json.load(f)
        # cached_eval_feature_file = os.path.join(args.data_dir, args.task_name, f"cached_eval_{args.max_len}")
        # # 判断缓存文件是否存在，如果存在，则直接加载处理后数据
        # if os.path.exists(cached_eval_feature_file):
        #     logger.info("已经存在eval缓存文件{}，直接加载".format(cached_eval_feature_file))
        #     eval_data_set = torch.load(cached_eval_feature_file)["data_set"]
        # # 如果缓存数据不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        # else:
        #     logger.info("不存在eval缓存文件{}，进行数据预处理操作".format(cached_eval_feature_file))
        #     eval_data_set = DialogueDataset(*process_bert(tokenizer, eval_data, args))
        #     logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_eval_feature_file))
        #     torch.save({"data_set": eval_data_set}, cached_eval_feature_file)
        # return train_data_set, eval_data_set, train_data, eval_data

    def build_input_data(self,example):
        knowledge, history, response = example["knowledge"],example["history"],example["response"]
        instance = dict()
        # sequence = [[ [self.bos] + knowledge ] + history + [response + [self.eos]]]
        # 三元组和开始标签在第一位，后面是历史对话，最后一句是回复
        # 在历史对话的最前面加入说话人的标签
        sequence = [[self.bos] + knowledge] + history + [response + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        # 最终得到 开始标记+三元组、带人物标记的历史对话和回复
        sequence = [sequence[0]] + sequence_with_speaker
        # 把所有列表拉成一个列表
        instance["input_ids"] = list(chain(*sequence))
        # 说话者的标记，与文本同长
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in
                                      s]
        #输入长度 - 1
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        #历史对话对应的值为 -100 ，回复的内容为原始值不变
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        return instance, sequence
    def __getitem__(self, item):
        _example = self.example[item]
        instance, _ = self.build_input_data(_example)
        return instance
    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        return input_ids, token_type_ids, lm_labels
    def __len__(self):
        return len(self.example)

def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays

def truncate_sequences(sequences, max_length):
    words_to_cut = sum(list(map(len, sequences))) - max_length
    if words_to_cut <= 0:
        return sequences

    while words_to_cut > len(sequences[0]):
        words_to_cut -= len(sequences[0])
        sequences = sequences[1:]

    sequences[0] = sequences[0][words_to_cut:]
    return sequences

# class ResonseDataset(torch.utils.data.Dataset):
#     def __init__(self,args):
#         super(ResonseDataset, self).__init__()
#         self.args = args
#         self.tokenizer = BertTokenizer.from_pretrained(self.args.vocab_path)
#         self.bos = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos_token"])
#         self.eos = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos_token"])
#         self.pad = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad_token"])
#         self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = \
#             self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["additional_special_tokens"])
#
#     def build_input(self, idx,sample):
#         knowledge, history, response = sample["knowledge"], sample["history"], sample["response"]
#         instance = dict()
#         sequence = [
#             [self.bos] + knowledge + history + [response]
#         ]
#         sequence_with_speaker = [
#             [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
#             for i, s in enumerate(sequence[1:])
#         ]
#         sequence = [sequence[0]] + sequence_with_speaker
#         instance["input_ids"] = list(chain(*sequence))
#         instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in
#                                       s]
#         instance["mc_token_ids"] = len(instance["input_ids"]) - 1
#         instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
#         return instance, sequence
#     def __getitem__(self, item):
#         return self.data_set[item]
#     def __len__(self):
#         return len(self.data_set)


##  duplicated


# class DialogueDataset(torch.utils.data.Dataset):
#     def __init__(self, history, knowledge, response, response_text, dialog_id):
#         self.history = history
#         self.knowledge = knowledge
#         self.response = response
#         self.response_text = response_text
#         self.dialog_id = dialog_id
#
#     def __getitem__(self, item):
#         instance = dict()
#         sequence = [args]
#
#
#
#         # return self.history[item], \
#         #        self.knowledge[item], \
#         #        self.response[item], \
#         #        self.response_text[item], \
#         #        self.dialog_id[item]
#
#     def __len__(self):
#         return len(self.history)
