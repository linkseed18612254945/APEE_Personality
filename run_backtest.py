import argparse
from langchain_openai import ChatOpenAI
from mbti_backtest import LLMBackTest
import os
from tqdm import tqdm
import json


def main(args):
    model = ChatOpenAI(model=args.model_name, temperature=0.1,
                       openai_api_base=args.api_base, openai_api_key=args.api_key)
    backtest = LLMBackTest()
    if os.path.isdir(args.test_answers_path):
        file_names = os.listdir(args.test_answers_path)
        file_paths = [os.path.join(args.test_answers_path, file_name) for file_name in file_names if file_name.endswith('answers.csv')]
        test_names = [file_name.replace('action_', 'action-').split('_')[-2] for file_name in file_names if file_name.endswith('answers.csv')]
    else:
        file_names = []
        file_paths = [args.test_answers_path]
        test_names = [args.test_answers_path.replace('action_', 'action-').split('_')[-2]]

    print(f"Start {args.test_answers_path} backtest,  with model {args.model_name}.")
    for file_path, test_name in tqdm(zip(file_paths, test_names), total=len(file_paths), desc=f'Backtest Files'):
        # try:
        save_path = file_path.replace('answers', 'backtest')
        if os.path.basename(save_path) in file_names:
            print(f"Skip {file_path}")
            continue
        back_test_result = backtest.backtest(model, file_path, test_name, args.max_concurrency)
        back_test_result.to_csv(save_path, index=False)
        print(f"Finnish backtest saved in {save_path}")
        # except Exception as e:
        #     print(e)


def build_args():
    parser = argparse.ArgumentParser()

    # Testing Model args
    parser.add_argument('--model_name', required=True, type=str, default='gpt-3.5-turbo-1106',
                        help='The name of the model to test')
    parser.add_argument('--api_base', required=True, type=str, default='https://api.chatanywhere.tech/v1',
                        help='fast-chat openai style api-base')
    parser.add_argument('--api_key', required=False, type=str,
                        default='sk-AwxSgUZaEPXxcxMCzSUpqhkzGLKjnLLfCCKSnUGQZFgztOPR',
                        help='fast-chat openai style api-key')
    parser.add_argument('--temperature', required=False, type=float, default=0.1,
                        help="LLM temperature")

    # Template args for testing Model
    parser.add_argument('--test_answers_path', required=True, type=str, default=0,
                        help='MBTI test answers path created by run_benchmark.py, back test all answers if the path is a dir')
    # Testing general args
    parser.add_argument('--max_concurrency', required=False, type=int, default=256,
                        help="max_concurrency for langchain calling apis")

    # Testing general args
    # parser.add_argument('--backtest_num', required=False, type=int, default=1000000,
    #                     help="max backtest num")
    return parser.parse_args()


if __name__ == '__main__':
    main(build_args())
