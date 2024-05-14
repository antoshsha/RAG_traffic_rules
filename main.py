import time
from collections import Counter
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
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
    """
    Display a prompt consisting of a question, options, and the correct answer.

    Parameters:
    - prompt (dict): A dictionary containing the components of the prompt.
                     It should have the following keys:
                     - 'question': (str) The question being asked.
                     - 'options': (list) A list of strings representing the options for the question.
                     - 'right_answer': (str) The correct answer to the question.
    """

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
    """
    Generate prompts for a given number of questions from a JSON file containing traffic rules questions and options.

    Parameters:
    - json_file (str): The path to the JSON file containing the questions and options.
    - number (int): The number of the document

    Returns:
    - prompts (list): A list of tuples, where each tuple contains the following elements:
                      1. The generated prompt (str).
                      2. The correct answer (str) for the prompt.
                      3. The number of the document (int).
                      4. The number of options for the question (int).
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    prompts = []

    prompt_template = Template("""
    You are an expert in Ukrainian traffic rules.
    I will ask you a question and provide options from which you should choose the right one.
    Send me only the text of the right option in Ukrainian.
    -----------
    Example:
    Вкажіть метод визначення причини різкого збільшення зусилля на кермовому колесі.
    Options:
    1.Візуальний огляд елементів системи гідропідсилювача рульового керування.
    2.Вимірювання робочого тиску в системі гідропідсилювача рульового керування.
    3.Діагностика під час руху.
    4.Варіанти 1 і 2.

    In this case your output must be "Варіанти 1 і 2." because both options 1 and 2 are correct
    -----------
    My question: {{ question }}
    Options:
    {% for option in options %}
    {{ loop.index }}. {{ option }}
    {% endfor %}
    ---------
    Send the answer as the text of the right option in Ukrainian. This is very important to my career.
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
    """
    Generate prompts for all JSON files in a given directory.

    Parameters:
    - directory (str): The path to the directory containing the JSON files.

    Returns:
    - all_prompts (list): A list of tuples, where each tuple contains the following elements:
                          1. The generated prompt (str).
                          2. The correct answer (str) for the prompt.
                          3. The number of the document (int).
                          4. The number of options for each question (int).
    """
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
    """
    Generate prompts for each topic represented by JSON files in a given directory.

    Parameters:
    - directory (str): The path to the directory containing the JSON files.

    Returns:
    - prompts_by_topic (list): A list of tuples, where each tuple contains the following elements:
                               1. The generated prompt (str).
                               2. The correct answer (str) for the prompt.
                               3. The number of the document (int).
                               4. The number of options for each question (int).
    """

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
    """
    Evaluate a list of prompts using semantic similarity embeddings.

    Parameters:
    - prompts (list): A list of tuples containing the prompts, correct answers, and other relevant information.
                      Each tuple should contain the following elements:
                      1. The generated prompt (str).
                      2. The correct answer (str) for the prompt.
                      3. Additional information if needed.
    - agent: The query_engine or agent used for generating responses.

    Returns:
    - None
    """
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
    """
    Evaluate prompts by topic with different query engines options.

    Parameters:
    - prompts (list): A list of tuples containing the prompts, correct answers, and other relevant information.
                      Each tuple should contain the following elements:
                      1. The generated prompt (str).
                      2. The correct answer (str) for the prompt.
                      3. The index of the topic (int).
                      4. The number of options for each question (int).
    - agents (list): A list of query engines or agents for generating responses.

    Returns:
    - None
    """
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
    """
    Evaluate prompts using GPT-3.5-turbo model for chat completions.

    Parameters:
    - prompts (list): A list of tuples containing the prompts, correct answers, and other relevant information.
                      Each tuple should contain the following elements:
                      1. The generated prompt (str).
                      2. The correct answer (str) for the prompt.
                      3. Additional information if needed.
    - client: The OpenAI client used for generating responses.

    Returns:
    - None
    """
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


topic_prompts = generate_prompts_by_topic("data/prompts")
prompts = generate_all_prompts("data/prompts")
sample_prompts = random.sample(prompts, 100)


# example of usage of evaluation with LLM
# llm_evaluator = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
# evaluate_prompts_llm(topic_prompts, llm_evaluator, query_engine)


# example of usage of separated RAG evaluation (multiple query engines)
asyncio.run(evaluate_prompts_by_topic_options(topic_prompts, list(doc_engines.values())))

# example of usage of regular RAG evaluation (one query engine)
asyncio.run(evaluate_prompts_by_topic_options(topic_prompts, [query_engine]))

# example of usage of LLM evaluation (second element in the array can be anything)
client = OpenAI()
asyncio.run(evaluate_prompts_by_topic_options(topic_prompts, [client, "gpt-3.5-turbo"]))












