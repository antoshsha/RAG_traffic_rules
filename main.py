import time
from collections import Counter
from dotenv import load_dotenv

from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from docs_loader import *
import numpy as np
from collections import defaultdict
from openai import OpenAI
import json
from jinja2 import Template
import asyncio
import random

load_dotenv()

random.seed(0)


def show_prompt(prompt):
    prompt_template = Template("""
    Question:
    {{question}}
    -----------
    Options:
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}
    -----------
    Right answer:
    {{right_answer}}
    """)
    question = prompt['question']
    options = prompt['options']
    right_answer = prompt['right_answer']
    print(prompt_template.render(question=question, options=options, right_answer=right_answer))


def generate_prompts(json_file, number):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    prompts = []

    prompt_template = Template("""
    Ти студент, що складає екзамен з правил дорожнього руху.
    Я поставлю тобі питання і запропоную варіанти відповіді і ти повинен обрати правильну відповідь.
    Надішли лише текст правильного варіанту відповіді українською мовою.
    -----------
    Приклад:
    Вкажіть метод визначення причини різкого збільшення зусилля на кермовому колесі.
    Варіанти відповіді:
    1.Візуальний огляд елементів системи гідропідсилювача рульового керування.
    2.Вимірювання робочого тиску в системі гідропідсилювача рульового керування.
    3.Діагностика під час руху.
    4.Варіанти 1 і 2.

    В цьому випадку твоя відповідь повинна бути: "Варіанти 1 і 2." тому що варіанти відповіді 1 і 2 обидва правильні.
    -----------
    Моє питання: {{ question }}
    Варіанти відповіді:
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}
    ---------
    Надішли лише текст правильного варіанту відповіді українською мовою. Це дуже важливо для моєї карʼєри.
    """)

    for prompt_id, prompt_data in data.items():
        question = prompt_data['question']
        options = prompt_data['options']
        number_of_options = len(options)
        right_answer = prompt_data['right_answer']
        rendered_prompt = prompt_template.render(question=question, options=options)
        prompts.append((rendered_prompt.strip(), right_answer, number, number_of_options))
    return prompts


def generate_all_prompts(directory):
    all_prompts = []
    file_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)
    file_paths = sorted(file_paths)
    for file_path in file_paths:
        prompts = generate_prompts(file_path, 1)
        all_prompts.extend(prompts)

    return all_prompts


def generate_prompts_by_topic(directory):
    prompts_by_topic = []
    file_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)
    file_paths = sorted(file_paths)
    for i in range(len(file_paths)):
        prompts = generate_prompts(file_paths[i], i)
        prompts_by_topic.extend(prompts)

    return prompts_by_topic


def evaluate_prompts_llm(prompts, llm_evaluator, agent):
    scores = []
    for i in range(len(prompts)):
        time.sleep(6)
        response = agent.query(prompts[i][0])
        print("right answer: ", prompts[i][1])
        print("responce: ", response)
        print("--------")
        evaluator = CorrectnessEvaluator(llm=llm_evaluator)
        eval_result = evaluator.evaluate(
            query=prompts[i][0],
            response=str(response),
            reference=prompts[i][1],
        )
        print(str(eval_result.score))
        print(str(eval_result.feedback))
        scores.append(eval_result.score)
    print("avg: ", np.mean(scores))
    print("median: ", np.median(scores))
    value_counts = Counter(scores)
    print("Value counts:")
    for value, count in value_counts.items():
        print(f"{value}: {count}")


async def evaluate_prompts_embeddings(prompts, agent):
    evaluator_embeddings = SemanticSimilarityEvaluator()
    avg_score = 0
    right_results = 0
    counter = 0
    for prompt in prompts:
        time.sleep(6)
        response = agent.query(prompt[0])
        print("system's answer:", response)
        print("right_answer: ", prompt[1])
        print("--------")
        result = await evaluator_embeddings.aevaluate(
            response=str(response),
            reference=prompt[1],
        )
        print(result.feedback)
        print(result.score)
        avg_score += result.score
        if result.score >= 0.97:
            right_results += 1
        counter += 1
        print("current_score: ", right_results / counter)
        print(counter)
    print("Avg score: ", avg_score / len(prompts))
    print("Right results ", right_results, "/", len(prompts))


async def evaluate_prompts_by_topic_options(prompts, agents):
    overall_counter = 0
    overall_right_answers = 0
    options_dict = defaultdict(int)
    evaluator_embeddings = SemanticSimilarityEvaluator()
    for i in range(33):
        print(f"evaluating topic {i + 1}")
        counter = 0
        right_answers = 0
        for prompt in prompts:
            if prompt[2] == i:
                counter += 1
                if len(agents) == 1:
                    response = agents[0].query(prompt[0])
                    time.sleep(6)
                elif len(agents) == 2:
                    response = agents[0].chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": f"{prompt[0]}"
                        }
                        ],
                        model="gpt-3.5-turbo-0613"
                    )
                    response = response.choices[0].message.content
                else:
                    time.sleep(6)
                    response = agents[i].query(prompt[0])
                print(response)
                print("right_answer: ", prompt[1])
                print("--------")
                result = await evaluator_embeddings.aevaluate(
                    response=str(response),
                    reference=prompt[1],
                )
                num_options = prompt[3]
                options_dict[f"overall{num_options}"] += 1
                print(result.score)
                if result.score > 0.97:
                    right_answers += 1
                    options_dict[f"right{num_options}"] += 1
        if counter != 0:
            print(f"accuracy for topic {i + 1} = {(right_answers / counter) * 100}%")
        overall_counter += counter
        overall_right_answers += right_answers
        print(f"two options: {options_dict['right2'] / options_dict['overall2']}%")
        print(f"three options: {options_dict['right3'] / options_dict['overall3']}%")
        print(f"four options: {options_dict['right4'] / options_dict['overall4']}%")
        print(f"five options: {options_dict['right5'] / options_dict['overall5']}%")
        print(f"overall accuracy: {(overall_right_answers / overall_counter) * 100}%")


async def evaluate_chat_gpt_3_5_turbo(prompts, client):
    evaluator_embeddings = SemanticSimilarityEvaluator()
    avg_score = 0
    right_results = 0
    for prompt in prompts:
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"{prompt[0]}"
            }
            ],
            model="gpt-3.5-turbo-0613"
        )
        response = response.choices[0].message.content
        print(response)
        print("--------")
        result = await evaluator_embeddings.aevaluate(
            response=str(response),
            reference=prompt[1],
        )
        print(result.feedback)
        print(result.score)
        avg_score += result.score
        if result.score >= 0.97:
            right_results += 1
    print("Avg score: ", avg_score / len(prompts))
    print("Right results ", right_results, "/", len(prompts))



client = OpenAI()
topic_prompts = generate_prompts_by_topic("data/prompts")
asyncio.run(evaluate_prompts_by_topic_options(topic_prompts, [client, "chat_gpt"]))












