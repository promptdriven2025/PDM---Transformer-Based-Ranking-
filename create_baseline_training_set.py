import re
import pandas as pd
import os
import pickle
from tqdm import tqdm
from run_ranking_E5 import rank_documents


def create_db_files(rel_rounds, state):
    if state == 'test':
        rel_pos = [2, 3, 4]
    else:
        rel_pos = [2, 4]

    create_summary = True
    summ_rows = []
    rel_rounds_string = ''.join([str(x) for x in rel_rounds])

    print(f"Creating {state} set for rounds {rel_rounds}")

    df = pd.read_csv('t_data.csv')
    rel_df = df[(df['position'].isin(rel_pos)) & (df['round_no'].isin(rel_rounds))]
    rows = list()

    for idx, row in tqdm(rel_df.iterrows(), total=len(rel_df)):
        top_df = df[(df.round_no == row.round_no) & (df.query_id == row.query_id) & (df.position < row.position)]
        d_cur = list(re.findall(r'.+?[.!?](?:\s+|$)', row["current_document"]))
        new_qid = str(row["query_id"]) + str(row.round_no) + str(int(row.position))
        new_qid = new_qid.rjust(5, '0')

        for idx_t, row_t in top_df.iterrows():
            g_pool = list(set(filter(lambda s: len(s.split()) >= 2,
                                     list(re.findall(r'.+?[.!?](?:\s+|$)', row_t["current_document"])))))
            for i in range(len(d_cur)):
                for j in range(len(g_pool)):
                    if len(d_cur[i].split()) < 2 or re.match(r'^\W*$', d_cur[i]):
                        continue
                    d_next = d_cur.copy()
                    d_next[i] = g_pool[j]
                    new_text = " ".join(d_next)
                    new_docno = row.docno + "$" + row_t.docno + "_" + str(i + 1) + "_" + str(j + 1)
                    temp_df = df[(df.round_no == row.round_no + 1) & (df.query_id == row.query_id)].copy()
                    new_row = [new_docno, new_text, row.round_no, float('nan'), "switched", row["query"],
                               float('nan'), float('nan')]
                    temp_df.loc[int(temp_df[temp_df.username == row.username].index[0])] = new_row
                    temp_df, embeddings_dict = rank_documents(row["query"], temp_df, "current_document",
                                                              return_embedding=True)
                    rank_promotion_label = max(0,
                                               row["position"] -
                                               temp_df[temp_df['username'] == 'switched']["rank"].values[
                                                   0])

                    if create_summary:
                        rank_promotion_true = row["position"] - \
                                              temp_df[temp_df['username'] == 'switched']["rank"].values[0]
                        summ_rows.append(
                            {"docno": new_docno, "rank": temp_df[temp_df['username'] == 'switched']["rank"].values[0],
                             "rank_promotion": rank_promotion_true,
                             "scaled_rank_promotion": rank_promotion_true / (
                                     row["position"] - 1) if rank_promotion_true > 0 else (
                                 rank_promotion_true / (
                                         temp_df['rank'].max() - row["position"]) if rank_promotion_true < 0 else 0),
                             "text": new_text})

                    embedding_values = " ".join([f"{key}:{value:.8f}" for key, value in embeddings_dict.items()])
                    output_line = f"{float(rank_promotion_label):.1f} qid:{new_qid} {embedding_values} # {new_docno}"
                    rows.append(output_line)

    with open(f"baseline_dataset_{state}_r{rel_rounds_string}.txt", "w") as f:
        f.write("\n".join(rows))
        print(f"Saved {len(rows)} lines to baseline_dataset_{state}_r{rel_rounds_string}.txt")

    if create_summary:
        pd.DataFrame(summ_rows).to_csv(f"baseline_dataset_{state}_r{rel_rounds_string}_summary.csv", index=False)
        print(f"Saved {len(summ_rows)} lines to baseline_dataset_{state}_r{rel_rounds_string}_summary.csv")


if __name__ == '__main__':
    round_state_list = [('test', [5, 6])]
    for state, rel_rounds in round_state_list:
        create_db_files(rel_rounds=rel_rounds, state=state)
