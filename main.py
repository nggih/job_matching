import glob

import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from utils import calculate_cosine_similarity, create_jd, get_text_from_resume

jd_links = ['https://datahack.analyticsvidhya.com/jobathon/clix-capital/senior-manager-growthrisk-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/clix-capital/manager-growth-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/clix-capital/manager-risk-analytics-2',
'https://datahack.analyticsvidhya.com/jobathon/cropin/data-scientist-85']


jd_df = create_jd(jd_links)

model = Doc2Vec.load('doc2vec.model')

resume_path = 'resume1.pdf'

resume_paths = glob.glob('./data/resumes/*.pdf')

results = []

for resume_path in resume_paths:
    container = {}
    resume = get_text_from_resume(resume_path=resume_path)
    container['resume_path'] = resume_path
    v1 = model.infer_vector(resume.split())

    # Loop through job description
    for i in range(len(jd_df['data'])):
        v2 = model.infer_vector(jd_df['data'][i].split())
        cosine_similarity = calculate_cosine_similarity(v1, v2)
        # Metrics
        container[i] = round(cosine_similarity, 3)

    results.append(container)

results_df = pd.DataFrame(results)
# based on jd number 4 (index=3) data scientist job, which resume is the best?
def get_best_3(results_df, idx=3):
    best_3 = results_df[idx].nlargest(3)
    best_df = results_df.iloc[best_3.index] 
    best_df = best_df[['resume_path', idx]]
    print(best_df)
    return best_df

get_best_3(results_df, 0)
get_best_3(results_df, 1)
get_best_3(results_df, 2)
get_best_3(results_df, 3)


