import json
import re
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm, trange
from langchain_core.output_parsers import StrOutputParser

import utils


class LLMBackTest:
    def __init__(self):
        self.mbti_keywords = json.load(open('data/mbti_keywords.json', 'r', encoding='utf8'))
        self.mbti_types = list(self.mbti_keywords.keys())
        self.output_example = """{"back_test_mbti": "XXXX", "analyse": "Answer the reason why you think the person\"s mbit is XXXX"}"""
        self.backtest_template = """
        Please analyse 4 axis MBTI trait 
        Introversion (I) – Extroversion (E)
        Intuition (N) – Sensing (S)
        Thinking (T) – Feeling (F)
        Judging (J) – Perceiving (P) step by step then predict the person's MBTI type 
        """
        self.output_parser = StrOutputParser()
        self.output_example = [
            {"I": 'analyse_prob in float number', "E": 'analyse_prob in float number'},
            {"S": 'analyse_prob in float number', "N": 'analyse_prob in float number'},
            {"T": 'analyse_prob in float number', "F": 'analyse_prob in float number'},
            {"J": 'analyse_prob in float number', "P": 'analyse_prob in float number'}
        ]

    def build_clean_chain(self, model):
        clean_template = """
        clean the json string to make sure the json_string can be decode correctly. the json string:
        {json_string}
        only output the correct json:
        """
        output_parser = StrOutputParser()
        clean_prompt = ChatPromptTemplate.from_template(clean_template)
        clean_chain = clean_prompt | model | output_parser
        return clean_chain

    def __answers_file_preprocess(self, answer_file_path):
        with open(answer_file_path) as f:
            answer_file = json.load(f)
        answers = answer_file['test_result']
        df = pd.DataFrame(answers)
        df['answer_without_mbti'] = df['answer'].apply(
            lambda x: re.sub('|'.join(map(re.escape, self.mbti_types)), "", x.lower()), '')
        return df

    def questionnaire_backtest(self, answer_file_path, clean_chain=None):
        answer_file = pd.read_csv(answer_file_path)
        answers = answer_file['q_a'].tolist()
        results = []
        for mbti_answer in tqdm(answers):
            mbti_answer = eval(mbti_answer)
            mbti_score = {
                'E': 0,
                'I': 0,
                'S': 0,
                'N': 0,
                'T': 0,
                'F': 0,
                'J': 0,
                'P': 0
            }
            valid = 0

            for q_a in mbti_answer:
                try:
                    response = q_a['answer']
                    # response = utils.json_parser(q_a['answer'], clean_chain)
                except Exception as e:
                    print(e)
                    response = None

                if response is not None and isinstance(response, dict) and 'answer' in response:
                    try:
                        choice = response.get('answer').upper()
                        if choice in ['A', 'B']:
                            mbti_choice = q_a['question'][choice]
                        else:
                            mbti_choice = choice
                        if mbti_choice in mbti_score:
                            mbti_score[mbti_choice] += 1
                        valid += 1
                    except Exception as e:
                        print(e)

            try:
                e_rate = round(mbti_score['E'] / (mbti_score['E'] + mbti_score['I']), 2)
                s_rate = round(mbti_score['S'] / (mbti_score['S'] + mbti_score['N']), 2)
                t_rate = round(mbti_score['T'] / (mbti_score['T'] + mbti_score['F']), 2)
                j_rate = round(mbti_score['J'] / (mbti_score['J'] + mbti_score['P']), 2)
                result = [
                    {'E': e_rate, "I": round(1 - e_rate, 2)},
                    {'S': s_rate, "N": round(1 - s_rate, 2)},
                    {'T': t_rate, "F": round(1 - t_rate, 2)},
                    {'J': j_rate, "P": round(1 - j_rate, 2)}
                ]
            except Exception as e:
                print(e)
                result = [
                    {'E': 0, "I": 0},
                    {'S': 0, "N": 0},
                    {'T': 0, "F": 0},
                    {'J': 0, "P": 0}
                ]
            results.append(result)
            # e_or_i = 'E' if mbti_score['E'] > mbti_score['I'] else 'I'
            # s_or_n = 'S' if mbti_score['S'] > mbti_score['N'] else 'N'
            # t_or_f = 'T' if mbti_score['T'] > mbti_score['F'] else 'F'
            # j_or_p = 'J' if mbti_score['J'] > mbti_score['P'] else 'P'
            # mbti_test_type = ''.join([e_or_i, s_or_n, t_or_f, j_or_p])
            # mbti_answer['back_test_mbti'] = mbti_test_type
            # mbti_answer['analyse'] = mbti_score
            # mbti_answer['valid_answer_rate'] = valid / len(mbti_answer['q_a'])
        answer_file['predict_mbti'] = results
        return answer_file

    @property
    def thinking_backtest_prompt(self):
        # thinking_template = """
        # according to the answer to the task in a scene.
        #
        # ### Scene Task
        # Scene: {scene}
        # The answer about what he would do or think about the [{task}] task in the above scene is:
        #
        # ### Answer
        # {answer}
        #
        # Please return decode-able json format with double quotation marks like {example}.
        # ### mbti and analyse
        # """
        # template = self.backtest_template + '\n' + thinking_template

        thinking_template = """
        Please analyse 4 axis MBTI trait, according to the person's answer to the task in a scene, analyze total four dimensions of the MBTI and give a possible score for each dimension
        ### Answer
        {answer}
        
        Output, analyse step by step, then given the judge and probability, 
        Extroversion(E) vs Introversion(I): 
        Sensing(S) vs Intuition(N):
        Thinking(T) vs Feeling(F):
        Judging(J) vs Perceiving(P):
        then final extract the analyse result in the format like {output_example}:
        """
        template = thinking_template

        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    @property
    def action_daily_backtest_prompt(self):
        action_daily_template = """     
        Please analyse 4 axis MBTI trait, according to the person's action  in social media daily, analyze total four dimensions of the MBTI and give a possible score for each dimension
        ### Answer
        {answer}
        
        Output, analyse step by step, then given the judge and probability, 
        Extroversion(E) vs Introversion(I): 
        Sensing(S) vs Intuition(N):
        Thinking(T) vs Feeling(F):
        Judging(J) vs Perceiving(P):
        then final extract the analyse result in the format like {output_example}:
        """
        template = self.backtest_template + '\n' + action_daily_template
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    @property
    def action_task_backtest_prompt(self):
        action_task_template = """
        Please analyse 4 axis MBTI trait, according to the person's action  to the task  in social media, analyze total four dimensions of the MBTI and give a possible score for each dimension
        ### Answer
        {answer}
        
        Output, analyse step by step, then given the judge and probability, 
        Extroversion(E) vs Introversion(I): 
        Sensing(S) vs Intuition(N):
        Thinking(T) vs Feeling(F):
        Judging(J) vs Perceiving(P):
        then final extract the analyse result in the format like {output_example}:
        """
        template = self.backtest_template + '\n' + action_task_template
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    @property
    def lgd_backtest_prompt(self):
        lgd_template = """
        Please analyse 4 axis MBTI trait, according to the person's role and conversation content in Leaderless Group Discussion(LGD), analyze total four dimensions of the MBTI and give a possible score for each dimension
        ### Task and Answer
        {answer}
        
        Output, analyse step by step, then given the judge and probability, 
        Extroversion(E) vs Introversion(I): 
        Sensing(S) vs Intuition(N):
        Thinking(T) vs Feeling(F):
        Judging(J) vs Perceiving(P):
        then final extract the analyse result in the format like {output_example}:
        """
        template = self.backtest_template + '\n' + lgd_template
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def build_backtest_prompt(self, test_type):
        if test_type == 'thinking':
            return self.thinking_backtest_prompt
        elif test_type == 'action-daily':
            return self.action_daily_backtest_prompt
        elif test_type == 'action-task':
            return self.action_task_backtest_prompt
        elif test_type == 'lgd':
            return self.lgd_backtest_prompt
        else:
            raise Exception(f"Wrong test type {test_type}")

    def backtest(self, model, answer_file_path, test_name='thinking', continue_backtest_path=None, max_concurrency=32):
        if test_name == 'questionnaire':
            clean_chain = self.build_clean_chain(model)
            back_test_result = self.questionnaire_backtest(answer_file_path, clean_chain)
            return back_test_result
        answers = pd.read_csv(answer_file_path)
        analyse_prompt = self.build_backtest_prompt(test_name)
        print(f"Analyse Num for {answer_file_path}: {answers.shape[0]}")
        chain = analyse_prompt | model | self.output_parser
        try:
            chain_args = [{"answer": answer, "output_example": self.output_example} for answer in
                          answers['answer'].tolist()]
            responses = chain.batch(chain_args, max_concurrency=max_concurrency)
        except Exception as e:
            print(e)
            responses = []
            for answer in tqdm(answers['answer'].tolist(), desc=f'Invoke'):
                chain_arg = {"answer": answer, "output_example": self.output_example}
                try:
                    response = chain.invoke(chain_arg)
                except Exception as e:
                    print(e)
                    response = ''
                responses.append(response)
        answers['back_test_response'] = responses
        predict_mbtis = []
        for r in responses:
            try:
                final_res = eval('[{' + r.split('[{')[1])
            except Exception as e:
                print(e)
                final_res = ''
            predict_mbtis.append(final_res)
        answers['predict_mbti'] = predict_mbtis
        return answers

    # def backtest(self, model, answer_file_path, test_name='thinking', continue_backtest_path=None, max_concurrency=32):
    #     clean_chain = self.build_clean_chain(model)
    #     if test_name == 'questionnaire':
    #         back_test_result = self.questionnaire_backtest(answer_file_path, clean_chain)
    #         return back_test_result
    #     with open(answer_file_path) as f:
    #         answer_file = json.load(f)
    #     answers = answer_file['test_result']
    #     analyse_prompt = self.build_backtest_prompt(test_name)
    #     chain = analyse_prompt | model | self.output_parser
    #     chain_args = [{arg: answer[arg] for arg in analyse_prompt.input_variables if arg in answer} for answer in answers]
    #     for chain_arg in chain_args:
    #         chain_arg['example'] = self.output_example
    #     print("Analyse Num: ", len(chain_args))
    #     responses = chain.batch(chain_args, max_concurrency=max_concurrency)
    #     for i, r in enumerate(responses):
    #         answers[i]['back_test_response'] = r
    #         try:
    #             final_res = eval('[{' + r.split('[{')[1])
    #         except Exception as e:
    #             print(e)
    #             final_res = ''
    #         answers[i]['predict_mbti'] = final_res
    #         # result = utils.json_parser(r, clean_chain)
    #         # answers[i].update(result)
    #     return answer_file

    def analyse(self, back_test_results):
        pass

