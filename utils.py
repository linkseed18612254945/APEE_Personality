import json
import re

import random
import string


def re_clean_json(json_string, catch_exception=True):
    # 首先转换外层单引号为双引号
    cleaned_str = json_string.strip().replace("'", '"')

    # 然后修复被错误替换的双引号（例如在 it's 中的情况）
    # 使用正则表达式仅保留字典格式正确的双引号
    cleaned_str = re.sub(r'"\s*:\s*"(.*?)"\s*(,|\})',
                         lambda m: '": "' + m.group(1).replace('"', "'") + '"' + m.group(2), cleaned_str)
    # 尝试将字符串转换为字典
    if catch_exception:
        try:
            data_dict = json.loads(cleaned_str)
            # 将字典转换为JSON格式的字符串
            # json_output = json.dumps(data_dict, indent=4, ensure_ascii=False)
            return data_dict
        except json.JSONDecodeError as e:
            print(f"Error converting to JSON: {str(e)}")
            return f"Error converting to JSON: {str(e)}"
    else:
        return json.loads(cleaned_str)

# def json_parser(response, clean_chain):
#     try:
#         result = json.loads(response)
#         return result
#     except Exception as e:
#         try:
#             result = re_clean_json(response, catch_exception=False)
#             return result
#         except Exception as e:
#             print(e)
#             return None

def remove_consecutive_duplicates(sentence):
    # 使用正则表达式找到连续的重复模式，并只保留一个
    pattern = re.compile(r'(```output\n)+')
    return pattern.sub(r'```output\n', sentence)

def remove_duplicate_sentences(paragraph):
    # 使用正则表达式分割段落为短句
    sentences = re.split(r'[.!?]', paragraph)

    seen_sentences = set()
    unique_sentences = []

    for sentence in sentences:
        # 移除首尾空格
        trimmed_sentence = sentence.strip()
        # 检查短句长度是否超过5个字符
        if len(trimmed_sentence.split()) > 5:
            # 检查是否已在结果中
            if trimmed_sentence not in seen_sentences:
                seen_sentences.add(trimmed_sentence)
                unique_sentences.append(trimmed_sentence)

    # 将处理过的短句重新组合为段落
    result_paragraph = '. '.join(unique_sentences) + '.'
    return result_paragraph




def json_parser(response, clean_chain):
    result = {}
    try:
        result = json.loads(response)
    except Exception as e:
        try:
            result = re_clean_json(response, catch_exception=False)
        except Exception as e:
            try:
                print(f"LLM Process Json")
                result = json.loads(clean_chain.invoke({"json_string": response}))
            except Exception as e:
                print(f"Error converting to JSON: {str(e)}")
    finally:
        return result

def generate_random_id(length=6):
    # 定义可能的字符集合，包括大写字母和数字
    characters = string.ascii_uppercase + string.digits
    # 从字符集合中随机选择字符，组成指定长度的字符串
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id