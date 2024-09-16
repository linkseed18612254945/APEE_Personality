import json
import random
import copy
import time

from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm, trange
from langchain_core.output_parsers import StrOutputParser
import utils


class Benchmark:
    def __init__(self):
        self.mbti_keywords = json.load(open('data/mbti_keywords.json', 'r', encoding='utf8'))
        self.characters_info = json.load(open('data/mbti_character_filter.json', 'r', encoding='utf8'))
        self.mbti_types = list(self.mbti_keywords.keys())
        self.output_parser = StrOutputParser()

    def build_prompt(self, template):
        raise NotImplementedError()

    def test(self, model, template, output_name, test_ratio, testing_role_nums=100000, max_history=20):
        raise NotImplementedError()

    # def save(self, save_res, output_name, type_name):
    #     save_path = f'output/{output_name}_{type_name}_answers.json'
    #     with open(save_path, 'w',
    #               encoding='utf-8') as f:
    #         json.dump(save_res, f)
    #     return save_path


class Questionnaire(Benchmark):
    def __init__(self):
        super().__init__()
        self.questions = json.load(open('data/questionnaire_en.json', 'r', encoding='utf8'))
        self.output_example = {"answer": "A", "thought": "Answer the reason why you choose A"}

    def build_prompt(self, template):
        question_template = """
        You are going to do the MBTI personality test, please answer the question according to your personality.
        ### Question
        {question} 
        
        Please return decode-able json format with double quotation marks and the content in 'thought' don't use double quotation marks like {example} 
        ### Answer and Thought 
        """
        full_template = template + '\n' + question_template
        prompt = ChatPromptTemplate.from_template(full_template)
        return prompt

    def __get_testing_result(
            self,
            chain,
            chain_args,
            test_ratio,
            batch=True,
            max_concurrency=256
    ):
        """
        get all answers of LLMs.
        """
        if max_concurrency > 1:
            batch = True
        else:
            batch = False
        q_a = []
        test_questions = random.sample(list(self.questions.values()), max(1, int(len(self.questions) * test_ratio)))
        if batch:
            inputs = []
            for q in test_questions:
                ca = copy.deepcopy(chain_args)
                ca['question'] = q
                inputs.append(ca) 
            responses = chain.batch(inputs, max_concurrency=max_concurrency)
            for question, response in zip(test_questions, responses):
                q_a.append({"answer": response, "question": question})

        else:
            for question in test_questions:
                chain_args['question'] = question
                response = chain.invoke(chain_args)
                q_a.append({"answer": response, "question": question})

        res = {
            'q_a': q_a
        }
        return res

    def test(self, model, template, output_name, test_ratio, testing_role_nums=100000, max_history=20, max_concurrency=256):
        prompt = self.build_prompt(template)
        chain = prompt | model | self.output_parser
        all_res = []

        for mbti_type in tqdm(self.mbti_types, desc="Testing MBTI Type"):
            keywords = ",".join(self.mbti_keywords[mbti_type].keys())
            traits = "\n".join(self.mbti_keywords[mbti_type].values())
            if 'biography' in prompt.input_variables or 'conversations' in prompt.input_variables:
                uids = list(self.characters_info[mbti_type].keys())[:testing_role_nums]
            else:
                uids = ['empty']  # Only Testing 1 time for no character setting
            for uid in tqdm(uids, desc='Testing User'):
                biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''

                chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits, "biography": biography,
                              "conversations": conversations, "example": self.output_example}
                res = self.__get_testing_result(chain, chain_args, test_ratio, max_concurrency=max_concurrency)
                res['mbti'] = mbti_type
                res['uid'] = uid
                all_res.append(res)
        return all_res


class SceneThinking(Benchmark):
    def __init__(self):
        super().__init__()
        with open('data/scene_thinking.json') as f:
            self.mbti_scenes = json.load(f)

    def build_prompt(self, template):
        thinking_template = """
        ### Scene
         {scene}
        
        What would you do or think about the {task} task in the above scene? do not mention your MBTI_TYPE. 
        ### Output
        """
        full_template = template + '\n' + thinking_template
        prompt = ChatPromptTemplate.from_template(full_template)
        return prompt

    def test(self, model, template, output_name, test_ratio=1, testing_role_nums=100000, max_history=20, max_concurrency=256):
        if max_concurrency > 1:
            batch = True
        else:
            batch = False
        prompt = self.build_prompt(template)
        chain = prompt | model | self.output_parser
        all_res = []

        for mbti_type in tqdm(self.mbti_types, desc="Testing MBTI Type"):
            keywords = ",".join(self.mbti_keywords[mbti_type].keys())
            traits = "\n".join(self.mbti_keywords[mbti_type].values())
            if 'biography' in prompt.input_variables or 'conversations' in prompt.input_variables:
                uids = list(self.characters_info[mbti_type].keys())[:testing_role_nums]
            else:
                uids = ['empty']  # Only Testing 1 time for no character setting

            if not batch:
                for uid in tqdm(uids, desc='Testing User'):
                    biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                    conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''
                    test_scenes = random.sample(self.mbti_scenes, max(1, int(len(self.mbti_scenes) * test_ratio)))
                    for scene_info in test_scenes:
                        scene = scene_info['Name'] + '\n' + scene_info['Context']
                        for task in list(scene_info['Tasks'].keys()):
                            chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits, "biography": biography,
                                          "conversations": conversations, "scene": scene, "task": task}
                            response = chain.invoke(chain_args)
                            res = {"mbti": mbti_type, "scene_name": scene_info['Name'], "scene": scene,
                                   "task": task, "answer": response, "uid": uid}
                            all_res.append(res)
            else:
                inputs = []
                mbti_res = []
                for uid in tqdm(uids, desc='Testing User'):
                    biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                    conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''
                    test_scenes = random.sample(self.mbti_scenes, max(1, int(len(self.mbti_scenes) * test_ratio)))
                    for scene_info in test_scenes:
                        scene = scene_info['Name'] + '\n' + scene_info['Context']
                        for task in list(scene_info['Tasks'].keys()):
                            chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits, "biography": biography,
                                        "conversations": conversations, "scene": scene, "task": task}
                            inputs.append(chain_args)
                            res = {"mbti": mbti_type, "scene_name": scene_info['Name'], "scene": scene,
                                "task": task, "answer": "", "uid": uid}
                            mbti_res.append(res)
                    try:
                        responses = chain.batch(inputs, max_concurrency=max_concurrency)
                    except Exception as e:
                        try:
                            responses = chain.batch(inputs, max_concurrency=max_concurrency)
                        except Exception as e:
                            responses = ["" for _ in range(len(inputs))]
                            print(e)

                    for i, r in enumerate(responses):
                        mbti_res[i]['answer'] = r
                    all_res.extend(mbti_res)
        return all_res


class SocialMediaAction(Benchmark):
    def __init__(self, daily_or_task='daily'):
        super().__init__()
        with open('data/social_media_actions.json') as f:
            self.actions = json.load(f)
        with open('data/social_media_tasks.json') as f:
            self.social_media_tasks = json.load(f)
        self.output_example = {
            "plan": [
                {
                    "action": "SearchingAndBrowsingContent",
                    "action_detail": "Searching what in detail",
                    "thought": "What you think about the reason you choose the action, and what is the contributiong to whole plan."
                },
                {
                    "action": "....",
                    "action_detail": "....",
                    "thought": "...."
                }
            ]
        }
        self.daily_or_task = daily_or_task

    def build_prompt(self, template):
        daily_action_template = """Now you need to role play social media use (like Twitter), the actions you can choose contains:
        ### Social Media Actions
        {actions}
        
        You are now asked to perform twenty daily social networking activities or other online behaviors 
        as realistically as possible, with a true sequential logic and chronological sequence between the behaviors.
        Do not mention your MBTI_TYPE, the output example should be like:  
        {example}
        ### Output
        """
        task_action_template = """Now you need to role play social media use (like Twitter), the actions you can choose contains:
        ### Social Media Actions
        {actions}
        
        ### Task
        What actions would you do in the task according to the your Personality:
        {task}
        
        Do not mention your MBTI_TYPE, the output example should be like:  
        {example}
        ### Output
        """
        action_template = daily_action_template if self.daily_or_task == 'daily' else task_action_template
        full_template = template + '\n' + action_template
        prompt = ChatPromptTemplate.from_template(full_template)
        return prompt

    def test(self, model, template, output_name, test_ratio, testing_role_nums=100000, max_history=20, max_concurrency=256):
        if max_concurrency > 1:
            batch = True
        else:
            batch = False
        prompt = self.build_prompt(template)
        chain = prompt | model | self.output_parser
        all_res = []

        for mbti_type in tqdm(self.mbti_types, desc="Testing MBTI Type"):
            keywords = ",".join(self.mbti_keywords[mbti_type].keys())
            traits = "\n".join(self.mbti_keywords[mbti_type].values())
            if 'biography' in prompt.input_variables or 'conversations' in prompt.input_variables:
                uids = list(self.characters_info[mbti_type].keys())[:testing_role_nums]
            else:
                uids = ['empty']  # Only Testing 1 time for no character setting

            if not batch:
                for uid in tqdm(uids, desc='Testing User'):
                    biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                    conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''
                    test_tasks_num = 2
                    test_tasks = random.sample(self.social_media_tasks, max(1, test_tasks_num))
                    # test_tasks = random.sample(self.social_media_tasks, max(1, int(len(self.social_media_tasks) * test_ratio)))
                    for task_info in test_tasks:
                        task = "TaskName: {}, Task Description: {}".format(task_info['Task'], task_info['Task Description'])
                        chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits, "biography": biography,
                                      "example": self.output_example,
                                      "conversations": conversations, "actions": self.actions, "task": task}
                        try:
                            response = chain.invoke(chain_args)
                        except Exception as e:
                            time.sleep(5)
                            try:
                                response = chain.invoke(chain_args)
                            except Exception as e:
                                response = ""
                                print(e)
                        res = {"daily_or_task": self.daily_or_task, "mbti": mbti_type,
                               "task": task if self.daily_or_task == 'task' else "empty",
                               "task_name": task_info['Task'], "answer": response, "uid": uid}
                        all_res.append(res)
            else:
                inputs = []
                mbti_res = []
                for uid in tqdm(uids, desc='Testing User'):
                    biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                    conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''
                    # test_tasks_num = int(len(self.social_media_tasks) * test_ratio)
                    test_tasks_num = 5
                    test_tasks = random.sample(self.social_media_tasks, max(1, test_tasks_num))
                    for task_info in test_tasks:
                        task = "TaskName: {}, Task Description: {}".format(task_info['Task'], task_info['Task Description'])
                        chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits, "biography": biography,
                                    "example": self.output_example,
                                    "conversations": conversations, "actions": self.actions, "task": task}
                        inputs.append(chain_args)
                        res = {"daily_or_task": self.daily_or_task, "mbti": mbti_type,
                            "task": task if self.daily_or_task == 'task' else "empty",
                            "task_name": task_info['Task'], "answer": "", "uid": uid}
                        mbti_res.append(res)
                try:
                    responses = chain.batch(inputs, max_concurrency=max_concurrency)
                except Exception as e:
                    try:
                        responses = chain.batch(inputs, max_concurrency=max_concurrency)
                    except Exception as e:
                        responses = ["" for _ in range(len(inputs))]
                        print(e)
                for i, r in enumerate(responses):
                    mbti_res[i]['answer'] = r
                all_res.extend(mbti_res)

        return all_res


# For Lgd Testing
class Participant:
    def __init__(self, participant_id, uid, chain, chain_args):
        self.participant_id = participant_id
        self.uid = uid
        self.chain = chain
        self.chain_args = chain_args

    def history_process(self, history, max_length=2048):
        history_str = ''
        for conv in history[::-1]:
            answer = conv['answer']
            speaker = conv['participant_id'] if conv['participant_id'] != self.participant_id else "yourself"
            history_str += f"{speaker}-{answer}\n"
        history_str = history_str[-max_length + 500:]
        return history_str

    def speak(self, topic, history):
        history_str = self.history_process(history)
        self.chain_args['history'] = history_str
        self.chain_args['topic'] = topic
        response = self.chain.invoke(self.chain_args)
        return response


def batch_speak(chain, participants, topics, historys, max_concurrency):
    inputs = []
    for p, t, h in zip(participants, topics, historys):
        history_str = p.history_process(h)
        chain_args = copy.deepcopy(p.chain_args)
        chain_args['history'] = history_str
        chain_args['topic'] = t
        inputs.append(chain_args)
    responses = []
    for i in trange(0, len(inputs), max_concurrency):
        sub_input = inputs[i: i + max_concurrency]
        try:
            sub_responses = chain.batch(sub_input, max_concurrency=max_concurrency)
        except Exception as e:
            sub_responses = ["" for _ in range(len(sub_input))]
            print(e)
        responses.extend(sub_responses)
    return responses


class LeaderlessGroupDiscussion(Benchmark):
    def __init__(self, group_size=6, discussion_turn=2, speak_chances_per_turn=3):
        super().__init__()
        with open('data/lgd_roles.json') as f:
            self.lgd_roles = json.load(f)
        with open('data/lgd_topics.json') as f:
            self.lgd_topics = json.load(f)
        self.output_example = {"role": "Choose role according to your mbti",
                               "content": "Your speech content according to your MBTI/role/discussion history, do not mention mbti type directly"}
        self.group_size = group_size
        self.discussion_turn = discussion_turn
        self.discussion_max_length = speak_chances_per_turn * group_size

    def build_prompt(self, template):
        lgd_template = """### Leaderless Group Discussion
        You are about to participate in a Leaderless Group Discussion(LGD), a unique collaborative exercise.
        As a participant, you are expected to actively engage in the discussion, contribute your ideas, listen to others, and help guide the group towards a meaningful conclusion or consensus on the topic provided.
        
        ### Roles
        There is no pre-assigned leader, but usually participants play one of a list of roles:
        {roles}
        
        ### LGD Topic
        Now you are asked to discuss the following topic:
        {topic}
        
        ### Discussion History
        {history}
        
        You should try to play a different role from others to show the uniqueness and value of your personality
        The output should contains the role you want to play according to your mbti trait then generate a new speech accordingly, like {example},You can choose the observer role if you don't want to speak. Make sure the output can be json parse correctly:
        ### Output
        """
        full_template = template + '\n' + lgd_template
        prompt = ChatPromptTemplate.from_template(full_template)
        return prompt

    def test(self, model, template, output_name, test_ratio, testing_role_nums=100000, max_history=20, max_concurrency=256):
        if max_concurrency > 1:
            batch = True
        else:
            batch = False
        prompt = self.build_prompt(template)
        chain = prompt | model | self.output_parser
        participants = []
        for mbti_type in self.mbti_types:
            keywords = ",".join(self.mbti_keywords[mbti_type].keys())
            traits = "\n".join(self.mbti_keywords[mbti_type].values())
            if 'biography' in prompt.input_variables or 'conversations' in prompt.input_variables:
                uids = list(self.characters_info[mbti_type].keys())[:testing_role_nums]
            else:
                uids = ['empty']  # Only Testing 1 time for no character setting

            for uid in uids:
                biography = "\n".join(self.characters_info[mbti_type][uid]['biography']) if uid != 'empty' else ''
                conversations = "\n".join(self.characters_info[mbti_type][uid]['conversations'][:max_history]) if uid != 'empty' else ''
                participant_id = utils.generate_random_id()

                chain_args = {"mbti_type": mbti_type, "keywords": keywords, "traits": traits,
                              "biography": biography, "conversations": conversations,
                              "example": self.output_example,
                              "roles": self.lgd_roles}

                participant = Participant(participant_id=participant_id, uid=uid, chain=chain, chain_args=chain_args)
                participants.append(participant)

        all_res = []
        topic_num = int(len(self.lgd_topics) * test_ratio)
        test_topics = random.sample(self.lgd_topics, max(1, topic_num))
        print(f"Batch {batch}")
        if not batch:
            for topic in test_topics:
                for turn in tqdm(range(self.discussion_turn), desc="Discussion turn"):
                    history = []
                    used_participants = random.choices(participants, k=self.group_size)
                    for chance in range(self.discussion_max_length):
                        choose_participant = used_participants[chance % len(used_participants)]
                        response = choose_participant.speak(topic, history)
                        res = {"topic": topic['name'], "mbti": choose_participant.chain_args['mbti_type'],
                            "participant_id": choose_participant.participant_id,
                            "answer": response, "uid": choose_participant.uid}
                        history.append(res)
                    all_res.append(history)
        else:
            for k in range(self.discussion_turn):
                historys = [[] for _ in range(len(test_topics))]
                for chance in range(self.discussion_max_length):
                    parts = []
                    for j, topic in enumerate(test_topics):
                        used_participants = random.choices(participants, k=self.group_size)
                        choose_participant = used_participants[chance % len(used_participants)]
                        parts.append(choose_participant)
                        res = {"topic": topic['name'], "mbti": choose_participant.chain_args['mbti_type'],
                               "participant_id": choose_participant.participant_id,
                               "answer": "", "uid": choose_participant.uid}
                        historys[j].append(res)
                    responses = batch_speak(chain, parts, test_topics, historys, max_concurrency)
                    for j in range(len(responses)):
                        historys[j][-1]['answer'] = responses[j]
                all_res.append(historys)
        return all_res
