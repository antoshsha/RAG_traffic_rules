# Application of RAG for Ukrainian Traffic Rules

## Overview
The objective of this work is to examine the power of RAG (Retrieval-Augmented Generation) in improving the knowledge of the model about Ukrainian traffic rules. Ukrainian traffic rules were selected as a representative example of legal documents. This project explores the effectiveness of RAG in enhancing the understanding and generation of responses related to Ukrainian traffic regulations.

## Dataset and Resources
The dataset of prompts and a collection of Markdown documents used for this project can be found [here](https://github.com/antoshsha/traffic_rules_questions_ua/).

## Project Structure
This repository contains the code for all RAG system creation, prompt creation and testing. It includes:
- RAG system creation in the `docs_loader.py` file
- Prompt generation: Creation of prompts for evaluation and testing purposes.
- Semantic similarity evaluation: Evaluation of generated responses using semantic similarity metrics.
- LLM (Language Model) judge evaluation: Evaluation of responses using a Language Model judge.
- Examples of usage: Demonstrations of how to utilize the functions provided in this repository.

### Note: you must have `.env` file in the main directory of the project with Open AI key and Cohere AI key.
