import json
import utils
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm, trange
import os
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1,
                   openai_api_base="https://api.chatanywhere.tech/v1", openai_api_key="sk-mFB7UssDEsa0xBlbembt9u4d4I8i5DdnJIDDJaGen8FwDqO2")
clean_template = """
clean the json string to make sure the json_string can be decode correctly. the json string:
{json_string}
only output the correct json:
"""
output_parser = StrOutputParser()
clean_prompt = ChatPromptTemplate.from_template(clean_template)
clean_chain = clean_prompt | model | output_parser


def clean_json(responses):
    parsed_responses = {}
    llm_parse_responses = []
    llm_parse_responses_index = []
    for i, response in enumerate(responses):
        result = None
        try:
            result = json.loads(response)
        except Exception as e:
            try:
                result = utils.re_clean_json(response, catch_exception=False)
            except Exception as e:
                pass
        if result is not None or len(response) > 16000:
            parsed_responses[i] = result
        else:
            llm_parse_responses.append({"json_string": response})
            llm_parse_responses_index.append(i)
    print(f"LLM Parse Json num: {len(llm_parse_responses)}, {len(llm_parse_responses) / len(responses)}")

    for i, response in tqdm(zip(llm_parse_responses_index, llm_parse_responses)):
        try:
            res = clean_chain.invoke(response)
            res = json.loads(res)
        except Exception as e:
            print(e)
            res = response
        parsed_responses[i] = res
    sorted_parsed_answers = [value for key, value in sorted(parsed_responses.items())]
    return sorted_parsed_answers


def clean_json_batch(responses, max_concurrency):
    parsed_responses = {}
    llm_parse_responses = []
    llm_parse_responses_index = []
    for i, response in enumerate(responses):
        result = None
        try:
            result = json.loads(response)
        except Exception as e:
            try:
                result = utils.re_clean_json(response, catch_exception=False)
            except Exception as e:
                pass
        if result is not None or len(response) > 16000:
            parsed_responses[i] = result
        else:
            llm_parse_responses.append({"json_string": response})
            llm_parse_responses_index.append(i)
    print(f"LLM Parse Json num: {len(llm_parse_responses)}, {len(llm_parse_responses)/len(responses)}")
    all_res = clean_chain.batch(llm_parse_responses, max_concurrency=max_concurrency)
    for i, res in zip(llm_parse_responses_index, all_res):
        try:
            parsed_responses[i] = json.loads(res)
        except Exception as e:
            print(e)
            parsed_responses[i] = res
    sorted_parsed_answers = [value for key, value in sorted(parsed_responses.items())]
    return sorted_parsed_answers


def question_clean(dir_path, output_dir):
    file_names = os.listdir(dir_path)
    output_names = os.listdir(output_dir)
    for name in tqdm(file_names):
        if '.json' not in name or name in output_names:
            continue

        answer_file_path = os.path.join(dir_path, name)
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_file = json.load(f)
        if answer_file['config_args']['template_name'] != 'role_profile_history':
            continue
        source_answers = answer_file['test_result']
        # if len(answers) > 1 or 'clean_answer' in answers[0]['q_a']:
        #     continue
        responses = []

        used_mbtis = []
        answers = []
        for mbti_answer in source_answers:
            if mbti_answer['mbti'] in used_mbtis:
                continue
            else:
                answers.append(mbti_answer)
                used_mbtis.append(mbti_answer['mbti'])
        for mbti_answer in answers:
            responses.extend([q_a['answer'] for q_a in mbti_answer['q_a']])
        try:
            clean_responses = clean_json(responses)
            index = 0
            for mbti_answer in answers:
                for i, q_a in enumerate(mbti_answer['q_a']):
                    mbti_answer['q_a'][i]['clean_answer'] = clean_responses[index]
                    index += 1
            with open(os.path.join(dir_path, 'cleaned', name), 'w') as f:
                json.dump(answer_file, f)
        except Exception as e:
            print(name)
            print(e)


def custom_join(texts, max_len=16000):
    # 将文本合并
    joined_text = '\n'.join(texts)
    # 如果合并后的文本长度不超过最大长度限制，直接返回
    if len(joined_text) <= max_len:
        return joined_text

    # 如果超过最大长度，找到最后一个结束标点符号的位置
    end_punctuation = ['.', '?', '!', '。', '？', '！']  # 列出可能的句子结束标点
    for i in range(max_len, 0, -1):
        if joined_text[i] in end_punctuation:
            return joined_text[:i + 1]
    return joined_text[:max_len]  # 如果没有合适的断点，仍然按最大长度截断


def extract_answer(answer_str):
    # 查找 "answer" 单词的位置
    answer_match = re.search(r'\banswer\b', answer_str)
    if not answer_match:
        raise ValueError("The input string does not contain the word 'answer'.")

    answer_pos = answer_match.start()

    # 查找 "A" 和 "B" 字母的位置
    a_positions = [m.start() for m in re.finditer(r'A', answer_str)]
    b_positions = [m.start() for m in re.finditer(r'B', answer_str)]

    # 合并并排序所有位置
    all_positions = a_positions + b_positions
    all_positions.sort()

    # 找到最近的 "A" 或 "B"
    closest_char = None
    min_distance = float('inf')
    for pos in all_positions:
        distance = abs(pos - answer_pos)
        if distance < min_distance:
            min_distance = distance
            closest_char = answer_str[pos]

    return {"answer": closest_char, "thought": ""}


def question_process(dir_path, output_dir, all_uids):
    file_names = os.listdir(dir_path)
    for name in tqdm(file_names):
        if '.json' not in name or 'question' not in name:
            continue
        save_path = os.path.join(output_dir, '_'.join(name.split('_')[1:]))
        save_path = save_path.replace('.json', '.csv')
        if os.path.isfile(save_path):
            continue
        answer_file_path = os.path.join(dir_path, name)
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_file = json.load(f)
        if answer_file['config_args']['template_name'] != 'role_profile_history':
            continue
        answers = answer_file['test_result']
        result = [qa for qa in answers if qa['uid'] in all_uids]
        for i, r in enumerate(result):
            qa_res = []
            for j, q_a in enumerate(r['q_a']):
                try:
                    q_a["answer"] = extract_answer(q_a['answer'])
                    qa_res.append(q_a)
                except Exception as e:
                    q_a["answer"] = {"answer": "A", "thought": ""}
                    qa_res.append(q_a)
            result[i]['q_a'] = qa_res
        result = pd.DataFrame(result)
        result.to_csv(save_path, index=False)
        print(result.shape[0])

def lgd_process(dir_path, output_dir):
    mbti_keywords = json.load(open('data/mbti_keywords.json', 'r', encoding='utf8'))
    mbti_types = [x.lower() for x in list(mbti_keywords.keys())]

    file_names = os.listdir(dir_path)
    output_names = os.listdir(output_dir)
    for name in tqdm(file_names):
        if '.json' not in name or 'lgd' not in name:
            continue
        save_path = os.path.join(output_dir, '_'.join(name.split('_')[1:]))
        save_path = save_path.replace('.json', '.csv')
        if os.path.isfile(save_path):
            continue
        answer_file_path = os.path.join(dir_path, name)
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            answer_file = json.load(f)
        answers = []
        if answer_file['config_args']['template_name'] != 'role_profile_history':
            continue
        for tr in answer_file['test_result']:
            for answer in tr:
                answers.extend(answer)
        answers = pd.DataFrame(answers)
        answers['answer'] = answers['answer'].apply(lambda x: x.lower())
        answers['answer'] = answers['answer'].apply(
            lambda x: re.sub('|'.join(map(re.escape, mbti_types)), "", x), '')
        answers['answer'] = "Topic:" + answers['topic'] + '  Talks: ' + answers['answer']
        result = answers.groupby('uid', group_keys=False).apply(filter_top_answers)
        result = result[['uid', 'mbti', 'answer']]
        # result = answers.groupby('uid').agg({
        #     'answer': lambda x: custom_join(x),
        #     'mbti': 'first'
        # })
        # result['uid'] = result.index
        result.to_csv(save_path, index=False)
        print(result.shape[0])

def filter_top_answers(group):
    # 找出前5个不同的answers
    top_answers = list(set(group['answer']))[:20]
    # 保留包含这些answers的行
    return group[group['answer'].isin(top_answers)]


def think_preprocess(dir_path, output_dir, all_uids):
    mbti_keywords = json.load(open('data/mbti_keywords.json', 'r', encoding='utf8'))
    mbti_types = [x.lower() for x in list(mbti_keywords.keys())]
    file_names = os.listdir(dir_path)

    for name in tqdm(file_names):
        if '.json' not in name or 'thinking' not in name:
            continue
        save_path = os.path.join(output_dir, '_'.join(name.split('_')[1:]))
        save_path = save_path.replace('.json', '.csv')
        if os.path.isfile(save_path):
            continue
        with open(os.path.join(dir_path, name), 'r', encoding='utf-8') as f:
            answer_file = json.load(f)
        if answer_file['config_args']['template_name'] != 'role_profile_history':
            continue
        else:
            answers = answer_file['test_result']
            answers = pd.DataFrame(answers)
            answers['scene'] = answers['scene'] + ".\n Task:" + answers['task']
            answers = answers[answers['uid'].apply(lambda x: x in all_uids)]
            answers = answers.sample(frac=1)
            answers['answer'] = answers['answer'].apply(lambda x: x.lower())
            answers['answer'] = answers['answer'].apply(
                lambda x: re.sub('|'.join(map(re.escape, mbti_types)), "", x), '')
            answers = answers.drop_duplicates(subset=['answer'])
            result = answers.groupby('uid', group_keys=False).apply(filter_top_answers)
            result['answer'] = result['scene'] + '\n Answer:' + result['answer']
            result = answers.groupby('uid').agg({
                'answer': lambda x: custom_join(x),  # 将answer字段拼接
                'mbti': 'first',
            })
            # result['uid'] = result.index
            result = result[['uid', 'mbti', 'answer']]

            result.to_csv(save_path, index=False)
            print(result.shape[0])


def action_preprocess(dir_path, output_dir, all_uids, task_or_daily='task'):
    mbti_keywords = json.load(open('data/mbti_keywords.json', 'r', encoding='utf8'))
    mbti_types = [x.lower() for x in list(mbti_keywords.keys())]
    file_names = os.listdir(dir_path)

    for name in tqdm(file_names):
        if '.json' not in name or 'action' not in name:
            continue
        save_path = os.path.join(output_dir, '_'.join(name.split('_')[1:]))
        save_path = save_path.replace('.json', '.csv')
        if os.path.isfile(save_path):
            continue
        with open(os.path.join(dir_path, name), 'r', encoding='utf-8') as f:
            answer_file = json.load(f)
        if answer_file['config_args']['template_name'] != 'role_profile_history':
            continue
        else:
            answers = answer_file['test_result']
            answers = pd.DataFrame(answers)
            answers = answers[answers['uid'].apply(lambda x: x in all_uids)]
            answers = answers.sample(frac=1)
            answers['answer'] = answers['answer'].apply(lambda x: str(x).lower())
            answers['answer'] = answers['answer'].apply(
                lambda x: re.sub('|'.join(map(re.escape, mbti_types)), "", x), '')
            answers = answers.drop_duplicates(subset=['answer'])
            result = answers.groupby('uid', group_keys=False).apply(filter_top_answers)
            if task_or_daily == 'task':
                result['answer'] = result['task'] + '\n Answers:' + result['answer']
            result = result[['uid', 'mbti', 'answer']]
            # result = answers.groupby('uid').agg({
            #     'answer': lambda x: custom_join(x),  # 将answer字段拼接
            #     'mbti': 'first',
            # })
            # result['uid'] = result.index
            result.to_csv(save_path, index=False)
            print(result.shape[0])


def process(dir_path, output_dir, use_uids):
    question_process(dir_path, output_dir, use_uids)
    think_preprocess(dir_path, output_dir, use_uids)
    action_preprocess(dir_path, output_dir, use_uids)
    lgd_process(dir_path, output_dir)


if __name__ == '__main__':
    output_dir = 'output/main_experiments'

    use_uid_info = pd.read_csv('data/use_uid.csv')
    use_uids = set(use_uid_info['uid'].tolist())
    question_process('output/questional', output_dir, use_uids)
    # process('output/chatgpt', output_dir, use_uids)
    # question_process('output/questional/cleaned', output_dir, use_uids)
    # think_preprocess('output/thinking', output_dir, use_uids)
    # action_preprocess('output/action_task', output_dir, use_uids)
    # action_preprocess('output/action_daily', output_dir, use_uids)
    # question_clean('output/questional/deepseek', 'output/questional/cleaned')
    lgd_process('output/lgd', output_dir)
