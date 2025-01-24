import glob
from collections import defaultdict
import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import utils
from model_train_test import run_command, run_commands
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")


def process_model(model, file_to_nick, archive_dir, model_dir, qrel_rankings):
    try:
        nick = file_to_nick[model]
        if "$" in model:
            print("ERROR! bot name contains $")

        for pos in ["2", "3", "4"]:  
            if os.path.exists(
                    f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/bot_followup_asrc_{nick}_{pos}.csv"):
                continue

            working_set_file_path = f'{archive_dir}/ws_output_{pos}.txt'

            features_file_path = f'{archive_dir}/features_{pos}.dat'
            predictions_file_path = f'/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/predictions_{nick}_{pos}.txt'

            command = f"/lv_local/home/user/jdk-21.0.1/bin/java -jar RankLib-2.18.jar -load {model_dir}/{model} -rank {features_file_path} -score {predictions_file_path}"

            run_bash_command(command)

            text_df = pd.DataFrame([line.strip().split(None, 2) for line in
                                    open(f'{archive_dir}/raw_ds_out_{pos}_texts.txt')],
                                   columns=['index_', 'ID', 'text'])
            text_df[['ref', 'docid']] = text_df['ID'].str.split('$', n=1, expand=True)
            text_df["creator"] = text_df["ref"].str.split("-", expand=True)[3]
            text_df["query_id"] = text_df["ref"].str.split("-", expand=True)[2].astype(int)

            df = pd.read_csv(working_set_file_path, delimiter=' ', header=None).sort_values([0, 2])
            score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])

            if int(score_column.isna().sum()) > 0:
                print(f"ERROR in {model}_{pos}: score column contains NaN values")
                continue

            df["score"] = score_column
            df['rank'] = df.groupby(0)['score'].rank(method='first', ascending=False).astype(int)
            df = df.rename(columns={0: 'qid', 2: 'docid'})[["qid", "docid", "score", "rank"]].sort_values(
                ['qid', 'docid'])

            df = df.merge(text_df, how='left', on='docid')
            df = df.merge(qrel_rankings, how='left', on='ID')
            df = df[df.true_rank.notna()]
            df['rank'] = df.groupby('qid')['score'].rank(method='first', ascending=False).astype(int)


            df_rank1 = df.query('rank == 1')

            df_rank1 = df_rank1[df_rank1.docid.str.contains("ROUND-04")]  
            df_rank1["round_no"] = "04"


            final_df = df_rank1
            final_df["username"] = "BOT_" + nick

            final_df = final_df[final_df['true_rank'] <= 4]

            final_df['true_rank_promotion'] = int(pos) - final_df['true_rank']

            final_df['true_rank_promotion_scaled'] = final_df.apply(
                lambda row: row['true_rank_promotion'] / (int(pos) - 1) if row['true_rank_promotion'] > 0 else
                row['true_rank_promotion'] / (4 - int(pos)) if row['true_rank_promotion'] < 0 else 0, axis=1)

            final_df.to_csv(
                f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/bot_followup_asrc_{nick}_{pos}.csv",
                index=False)
        return (model, True)
    except Exception as e:
        print(e)
        return (model, False)


if __name__ == '__main__':

    run_command('find /lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/ -type f -exec rm {} +')
    archive_dir = "/lv_local/home/user/train_fb_ranker/archive_test_w2v"

    qrel_rankings = pd.read_csv("feature_data_asrc_t.csv")
    qrel_rankings = qrel_rankings[qrel_rankings.docno.str.contains('ROUND')]
    qrel_rankings = qrel_rankings.rename({'docno': 'ID', 'rank': 'true_rank'}, axis=1)
    qrel_rankings['ID'] = qrel_rankings['ID'].apply(lambda x: '-'.join(x.split("-")[2:-1]))
    qrel_rankings = qrel_rankings[['ID', 'true_rank']]

    model_dir = "/lv_local/home/user/train_fb_ranker/trained_models"
    file_to_nick = {k: k.split("_")[-1] for k in os.listdir(model_dir) if "model" in k}
    models = ['harmonic1_TESTmodel_LMBOT1']

    print(f'model number {len(models)}\n\n')

    max_workers = os.cpu_count() - 10
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_model, model, file_to_nick, archive_dir, model_dir, qrel_rankings) for model
                   in models]

        for future in tqdm(futures, total=len(models)):
            result = future.result()

    df = pd.concat(
        [pd.read_csv(file) for file in
         glob.glob("/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/bot_followup_asrc_*.csv")],
        ignore_index=True).sort_values(["round_no", "query_id", "creator"])

    df.to_csv(
        f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/bot_followup_asrc_test_FULL.csv",
        index=False)

    df = df[['username', 'true_rank', 'true_rank_promotion', 'true_rank_promotion_scaled']].groupby(
        'username').mean().sort_values('true_rank_promotion_scaled', ascending=False)

    if not df.empty:
        df.to_csv("/lv_local/home/user/train_fb_ranker/output_results/best_test_bots.csv",
                  index=True)
