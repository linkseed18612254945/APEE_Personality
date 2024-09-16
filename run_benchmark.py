import argparse
from langchain_openai import ChatOpenAI
from ppeval import *
import utils


def save(save_res, output_name, type_name):
    save_path = f'output/{utils.generate_random_id(6)}_{output_name}_{type_name}_answers.json'
    with open(save_path, 'w',
              encoding='utf-8') as f:
        json.dump(save_res, f)
    return save_path


def main(args):
    print(vars(args))
    with open('templates.json') as f:
        templates = json.load(f)

    assert 0 < args.test_ratio <= 1, f"test_ratio {args.test_ratio} must in (0, 1]"
    assert args.template_name in templates.keys(), f"{args.template_name} not in {';'.join(templates.keys())}"

    model = ChatOpenAI(model=args.model_name, temperature=0.1,
                       openai_api_base=args.api_base, openai_api_key=args.api_key)
    template = templates.get(args.template_name)

    if args.test_name == 'questionnaire':
        test_benchmark = Questionnaire()
    elif args.test_name == 'thinking':
        test_benchmark = SceneThinking()
    elif args.test_name == 'action_daily':
        test_benchmark = SocialMediaAction(daily_or_task='daily')
    elif args.test_name == 'action_task':
        test_benchmark = SocialMediaAction(daily_or_task='task')
    elif args.test_name == 'lgd':
        test_benchmark = LeaderlessGroupDiscussion(args.group_size, args.discussion_turn, args.speak_chances_per_turn)
    else:
        raise Exception(f"test_name {args.test_name} not in questionnaire/thinking/action_daily/action_task/lgd")
    print(f"Start {args.test_name} testing, ratio {args.test_ratio}, use template {args.template_name} with model {args.model_name}.")
    result = test_benchmark.test(model, template, output_name=args.model_name, test_ratio=args.test_ratio,
                                 testing_role_nums=args.testing_role_nums, max_history=args.max_history, max_concurrency=args.max_concurrency)
    save_result = {
        "config_args": vars(args),
        "test_result": result
    }
    save_path = save(save_result, args.model_name, args.test_name)
    print(f"Finnish test, result saved in {save_path}")


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
    parser.add_argument('--template_name', required=True, type=str,
                        help='Model prompt template,  available in templates.json')
    parser.add_argument('--test_name', required=True, type=str, default=0,
                        help='MBTI Test types: questionnaire/thinking/action_daily/action_task/lgd')
    parser.add_argument('--target_mbti', required=False, type=str, default=None,
                        help='Only Test target mbti type')

    # Testing general args
    parser.add_argument('--test_ratio', required=False, type=float, default=1,
                        help="Testing ratio for any benchmark 0<ratio<=1 ")
    parser.add_argument('--testing_role_nums', required=False, type=int, default=100000,
                        help="Testing distinct characters for each mbti type, default is all")
    parser.add_argument('--max_history', required=False, type=int, default=10,
                        help="Character's max conversation history length")
    parser.add_argument('--max_concurrency', required=False, type=int, default=256,
                        help="max_concurrency for langchain calling apis")

    # LGD(leaderless group discussion) args
    parser.add_argument('--group_size', required=False, type=int, default=6,
                        help="Discussion group size")
    parser.add_argument('--discussion_turn', required=False, type=int, default=2,
                        help="Discussion turn for each topic")
    parser.add_argument('--speak_chances_per_turn', required=False, type=int, default=3,
                        help="Speach chances for each participant in a single discussion turn")
    return parser.parse_args()


if __name__ == '__main__':
    main(build_args())