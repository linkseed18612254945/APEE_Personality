<p align="center">
  <img center src="https://i.postimg.cc/C5zYG9B1/20240929225943.png" width = "150" alt="logo">
</p>

<h2 align="center">APEE - Assessing the Personality Expressions of LLM-driven Role Play
Agent</h2>

## Table of Contents

- [Overview](#overview)
- [Installation](#Installation)
- [Download](#Download)

## Overview
**APEE** is a new benchmark consisting of 473 instances across three real-world scenario types: practical goal planning, social media behavior, and leaderless group discussions.

## Installation

Requires Python 3.10 to run.

Install conda environment from `environment.yml` file.

```sh
conda env create -n finfact --file environment.yml
conda activate finfact
```

## Download
The dataset can be found in the **data**  directory.

## Dataset Description
- **questionnaire_en.json**: The commonly used MBTI questionnaire includes 70 questions along with corresponding options.
- **scene_thinking.json**: The data used to assess personality traits when RPAs encounters reasoning problems in real-world scenarios.
- **social_media_tasks.json**: The data used to evaluate how RPA's behavior choices on social media reflect personality traits.
- **social_media_actions.json**: Behaviors available for RPA in social media.
- **lgd_topics.json**: Topics for leaderless group discussions used to assess the personality traits of RPAs in communication dialogues.
- **lgd_roles.json**: Self-positioning options available for RPAs in leaderless group discussions.
- **mbti_keywords.json**: A detailed description of the 16 MBTi personality traits.
- **mbti_character.json**: Characters that LLMs need to role-play, including character profiles, MBTI types, and historical dialogues, collected from https://www.personality-database.com/.
