# job_matching

Job Matching Similarity Analysis

## Dataset

Problem: Job Matching Similarity Analysis
Description: Your task is to design and implement a job-matching similarity analysis program that analyzes the similarity between resumes and job postings based on their attributes. You are free to choose the attributes that you believe are relevant for calculating the similarity. Additionally, you should find and use a suitable dataset of resumes and job postings for testing and evaluation purposes. Please store the dataset on GitHub and provide clear instructions on how to access and use the dataset.
Requirements:

1. Design and implement a solution that calculates the similarity between resumes and job postings based on their attributes.

- Choose the relevant attributes that you believe contribute to the similarity calculation.
- Define a suitable similarity measure or metric that takes into account the selected attributes.

2. Find and use a suitable resumes and job postings datasets for testing and evaluation.

- Store the dataset on GitHub in a format easily accessible and used by the interviewers.
- Include clear instructions in your submission on accessing and using the dataset.

3. Implement a Python program that performs the following tasks:

- Load the resumes and job postings dataset.
- Provide a function or method to calculate the similarity between resumes and job postings.
- Provide a function or method to find the most similar jobs for a given resume.
- Display the results meaningfully, showing the top N most similar job postings for resumes and their similarity scores.

4. Write clear instructions in your submission on how to run and test the program.

- Specify any dependencies or libraries required to run the program.
- Explain the steps to load the dataset, run the similarity analysis, and interpret the results.
  Deliverables:
- A Python program or package that implements the job matching similarity analysis functionality.
- The dataset of resumes and job postings are stored on GitHub, along with clear instructions on accessing and using it.
- Documentation or a readme file explaining the approach, design choices, and instructions for running and testing the program.

Evaluation Criteria:

- Design and implementation of the job matching similarity analysis solution.
- Accuracy and effectiveness of the chosen similarity measure and attributes.
- Proper storage and accessibility of the dataset on GitHub.
- Clarity and completeness of the instructions for running and testing the program.
- The overall quality of the code and documentation.
  Please make sure to provide all the necessary information and instructions for the interviewers to evaluate and test your solution effectively. Good luck with your assignment!

## SOLUTION:

1. using Doc2Vec by building the model with a relevant corpus (skill based wordings.
2. build a job description 'database', vectorized it
3. vectorized resume
4. compare the cosine similarity of resume's vector and the each of the job description's vector
5. get the best 3

## Installation

```code
pip install -r requirements.txt
```

## Run

```code
python model_builder.py
```

## Then

```code
python main.py
```
