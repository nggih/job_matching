
import re

import numpy as np
import pandas as pd
import PyPDF2
import requests
from bs4 import BeautifulSoup


def extract_data(url):
    list1 = []
    count = 0
    resp = requests.get(url)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text,'html.parser')
        l = soup.find(class_ = 'av-company-description-page mb-2')
        web = ''.join([i.text for i in l.find_all(['p', 'li'])])
        list1.append(web)
        return web
    else:
        print("Error")

def calculate_cosine_similarity(v1, v2):
    cosine_similarity = (np.dot(np.array(v1), np.array(v2))) / (np.linalg.norm(np.array(v1)) * np.linalg.norm(np.array(v2)))
    return cosine_similarity

def create_jd(jd_links):
    jd_df = pd.DataFrame(columns = ['links', 'data'])
    jd_df['links'] = jd_links

    for i in range(len(jd_df)):
        jd_df['data'][i] = extract_data(jd_df['links'][i])

    #Converting the text into lower case
    jd_df.loc[:,"data"] = jd_df.data.apply(lambda x : str.lower(x))

    #Removing the punctuations from the text
    jd_df.loc[:,"data"] = jd_df.data.apply(lambda x : " ".join(re.findall('[\w]+',x))
    )
    #Removing the numerics present in the text
    jd_df.loc[:,"data"] = jd_df.data.apply(lambda x : re.sub(r'\d+','',x))
    return jd_df

def get_text_from_resume(resume_path):
    resume = ''
    pdf_reader = PyPDF2.PdfReader(resume_path)
    for page in pdf_reader.pages:
        resume += page.extract_text()

    resume = resume.lower()
    resume = re.sub('[^a-z]', ' ', resume)
    return resume

def single_resume_compare_to_jd_db(model, jd_df, resume_path):
    print(resume_path)
    resume = get_text_from_resume(resume_path=resume_path)
    v1 = model.infer_vector(resume.split())

    # Loop through job description
    for i in range(len(jd_df['data'])):
        v2 = model.infer_vector(jd_df['data'][i].split())
        cosine_similarity = calculate_cosine_similarity(v1, v2)
        # Metrics
        print(i, round(cosine_similarity, 3))
