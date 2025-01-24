import pandas as pd
from run_ranking_E5 import rank_documents
from tqdm import tqdm

relevant_base_round = 5
summ_rows = []

df = pd.read_csv('t_data.csv')
old_df = df[df['round_no'] == relevant_base_round][["query_id", "username", "position"]].rename(
    {"position": "original_pos"}, axis=1)
new_df = df[df['round_no'] == relevant_base_round + 1]

bot_followup = pd.read_csv(
    f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/LMBOT1_files/bot_followup_t{relevant_base_round}@LMBOT1.csv")

query_legend = pd.read_csv("query_legend.csv")

dfs = list()

for idx, row in tqdm(bot_followup.iterrows()):
    leg_df = new_df[(new_df.query_id == row.query_id) & (new_df.username != row.creator)]
    leg_df = pd.concat([leg_df, row.to_frame().T.rename({'text': 'current_document'}, axis=1)])
    query_text = query_legend[query_legend.query_id == row.query_id]["query"].iloc[0]
    leg_df = rank_documents(query_text, leg_df, "current_document", return_embedding=False)
    merged_df = leg_df[leg_df.username != "LMBOT1"].dropna(how='all', axis=1).merge(old_df, on=['query_id', 'username'],
                                                                                    how='left')
    merged_df['rank_promotion'] = merged_df['original_pos'] - merged_df['rank']
    merged_df['scaled_rank_promotion'] = merged_df.apply(
        lambda row: row['rank_promotion'] / (row['original_pos'] - 1)
        if row['rank_promotion'] > 0 else (
            row['rank_promotion'] / (leg_df['rank'].max() - row['original_pos'])
            if row['rank_promotion'] < 0 else 0),
        axis=1
    )
    merged_df["query_id_full"] = str(relevant_base_round + 1) + str(int(row.query_id)).rjust(3, '0') + row.creator
    merged_df = merged_df.rename(
        {"rank": "current_pos", "query_id": "qid", "original_pos": "previous_pos", "rank_promotion": "pos_diff",
         "scaled_rank_promotion": "scaled_pos_diff"}, axis=1)
    dfs.append(merged_df[["query_id_full", "docno", "current_pos", "qid", "username", "previous_pos", "pos_diff",
                          "scaled_pos_diff"]])

student_df = pd.concat(dfs)
student_df.to_csv(
    f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/LMBOT1_files/students_t{relevant_base_round}@LMBOT1.csv",
    index=False)
