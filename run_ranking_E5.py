import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import xml.etree.ElementTree as ET
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-unsupervised')
model = AutoModel.from_pretrained('intfloat/e5-large-unsupervised').to(device)


def get_texts_from_trectext(file_path):
    data = []
    with open(file_path, 'r') as file:
        docno = ''
        text = ''
        for line in file:
            if '<DOCNO>' in line:
                docno = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
            elif '<TEXT>' in line:
                text = line.replace('<TEXT>', '').strip()
            elif '</TEXT>' in line:
                text += ' ' + line.replace('</TEXT>', '').strip()
                data.append([docno, text])
            elif '<DOC>' in line or '</DOC>' in line:
                continue
            else:
                text += ' ' + line.strip()
    return pd.DataFrame(data, columns=["docno", "text"])


def get_working_set(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) > 2:
                qid = parts[0]
                docno = parts[2]
                data.append([qid, docno])
    return pd.DataFrame(data, columns=["qid", "docno"])


def get_queries_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for query in root.findall('query'):
        qid = query.find('number').text
        query_str = query.find('text').text.replace('#combine(', '').replace(')', '')
        data.append([qid, query_str])
    return pd.DataFrame(data, columns=["qid", "query_str"])


def compute_embeddings(texts):
    inputs = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)


def rank_documents(query, bot_followup_df, text_col='text', return_embedding=False):
    bot_followup_df = bot_followup_df.reset_index(drop=True)
    with torch.no_grad():
        input_texts = [f'query: {query}'] + [f'passage: {d}' for d in bot_followup_df[text_col].tolist()]
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores = pd.Series(scores.tolist()[0])
        bot_followup_df['score'] = scores
        bot_followup_df['rank'] = bot_followup_df['score'].rank(ascending=False, method='min')

        if return_embedding:
            rel_doc = bot_followup_df[bot_followup_df['username'] == 'switched'][text_col].values[0]
            inputs = {key: value.to(device) for key, value in
                      tokenizer(rel_doc, return_tensors="pt", truncation=True, padding=True, max_length=512).items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
            embeddings_dict = {i + 1: embeddings[i] for i in range(len(embeddings))}
            return bot_followup_df, embeddings_dict

        return bot_followup_df


def main(name):
    input_dir = './input_files'
    bot_followup_file = f'{input_dir}/bot_followup_{name}.trectext'
    working_set_file = f'{input_dir}/working_set_{name}.trectext'
    queries_file = f'{input_dir}/queries_{name}.xml'

    bot_followup_df = get_texts_from_trectext(bot_followup_file)
    working_set_df = get_working_set(working_set_file)
    queries_df = get_queries_from_xml(queries_file)

    ranking_df = working_set_df.merge(bot_followup_df, on='docno', how='left').merge(queries_df, on='qid', how='left')
    query_dict = ranking_df[['qid', 'query_str']].drop_duplicates().set_index('qid').to_dict('index')

    dfs = []
    for qid, df in tqdm(ranking_df.groupby('qid'), desc='Ranking documents', total=len(query_dict)):
        query_str = query_dict[qid]['query_str']
        new_df = rank_documents(query_str, df)
        dfs.append(new_df)

    final_df = pd.concat(dfs).reset_index(drop=True)
    final_df = final_df.sort_values(by=['qid', 'rank'])
    final_df.to_csv(f'output_files/feature_data_{name}.csv', index=False)


if __name__ == "__main__":
    cp = "t4@F200"
    main(cp)