import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


def process_model(model_name, state, model_dir, features_file_path, df, return_texts_df=False):
    predictions_file_path = f'/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/predictions_{model_name}_{state}.txt'

    if not os.path.exists(predictions_file_path):
        command = f"/lv_local/home/user/jdk-21.0.1/bin/java -jar RankLib-2.18.jar -load {model_dir}/{model_name} -rank {features_file_path} -score {predictions_file_path}"
        output = run_bash_command(command)

    score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])

    if int(score_column.isna().sum()) > 0:
        print(f"ERROR in {model_name}: score column contains NaN values")

    df["score"] = score_column
    df['position'] = df.groupby("orig_docno")['score'].rank(method='first', ascending=False).astype(int)
    if return_texts_df:
        return df
    res_dict = df[df.position == 1][['rank', 'rank_promotion', 'scaled_rank_promotion']].mean().to_dict()
    res_dict["model"] = model_name.split("_")[-1]

    return res_dict


def process_top_model(model_name, path_dict, state):
    df = pd.read_csv(path_dict["summary"])
    orig_df = pd.read_csv("t_data.csv")[["docno", "current_document", "position"]]
    df["orig_docno"] = df.docno.apply(lambda x: x.split("$")[0])
    predictions_file_path = f'/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/predictions_baseline_model_{model_name}_{state}.txt'
    score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])
    if int(score_column.isna().sum()) > 0:
        print(f"ERROR in {model_name}: score column contains NaN values")

    df["score"] = score_column
    df['position'] = df.groupby("orig_docno")['score'].rank(method='first', ascending=False).astype(int)
    df = df[df.position == 1].reset_index(drop=True).drop("position", axis=1)
    df[["round_no", "query_id", "creator"]] = df.orig_docno.apply(
        lambda x: pd.Series([int(x.split("-")[1]), int(x.split("-")[2]), x.split("-")[3]]))
    df["username"] = "LMBOT1"
    df = df.merge(orig_df, left_on="orig_docno", right_on="docno", how="left").rename(
        columns={"current_document": "ref_doc"})

    test_round = df.round_no.iloc[0]

    bot_followup = df[["round_no", "query_id", "creator", "username", "text"]]
    bot_followup.to_csv(
        f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/LMBOT1_files/bot_followup_t{test_round}@LMBOT1.csv",
        index=False)

    feature_data = df[["query_id", "orig_docno", "rank", "score"]].rename(columns={"orig_docno": "docno"})
    feature_data.to_csv(
        f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/LMBOT1_files/feature_data_t{test_round}@LMBOT1.csv",
        index=False)

    feature_data_new = df[
        ["round_no", "query_id", "creator", "username", "orig_docno", "position", "rank", "rank_promotion",
         "scaled_rank_promotion", "score"]].rename(
        columns={"orig_docno": "docno", "position": "original_position", "rank": "current_pos",
                 "rank_promotion": "pos_diff", "scaled_rank_promotion": "scaled_pos_diff"})
    feature_data_new.to_csv(
        f"/lv_local/home/user/train_fb_ranker/output_results/ranker_test_results/LMBOT1_files/feature_data_new_t{test_round}@LMBOT1.csv",
        index=False)


def main(path_dict, state):
    df = pd.read_csv(path_dict["summary"])
    df["orig_docno"] = df.docno.apply(lambda x: x.split("$")[0])
    models = [model for model in os.listdir("/lv_local/home/user/train_fb_ranker/trained_models") if
              "baseline_model" in model]

    rows = []

    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(process_model, model_name, state, "trained_models", path_dict["embeddings"],
                                           df[["docno", "orig_docno", "rank", "rank_promotion",
                                               "scaled_rank_promotion"]]): model_name for model_name in models}

        for future in tqdm(as_completed(future_to_model), total=len(models)):
            model_name = future_to_model[future]
            try:
                res_dict = future.result()
                rows.append(res_dict)
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')

    res_df = pd.DataFrame(rows)
    res_df.sort_values("scaled_rank_promotion", ascending=True).to_csv(
        f"/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_{state}.csv",
        index=False)
    return res_df


if __name__ == '__main__':
    test_paths = {"embeddings": "baseline_dataset_test_r56.txt", "summary": "baseline_dataset_test_r56_summary.csv"}
    val_paths = {"embeddings": "baseline_dataset_validation_r4.txt",
                 "summary": "baseline_dataset_validation_r4_summary.csv"}

    round_no = test_paths['embeddings'].split("_r")[1].split(".")[0]

    for path in list(test_paths.values()) + list(val_paths.values()):
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist")
            exit(1)
    if not os.path.exists(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_test.csv"):
        res_df_test = main(test_paths, "test")
    else:
        res_df_test = pd.read_csv(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_test.csv")

    if not os.path.exists(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_val.csv"):
        res_df_val = main(val_paths, "val")
    else:
        res_df_val = pd.read_csv(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_val.csv")

    if not os.path.exists(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_full.csv"):
        res_df_full = pd.merge(res_df_val, res_df_test, on="model", suffixes=("_val", "_test")).sort_values(
            ["scaled_rank_promotion_val", "scaled_rank_promotion_test"], ascending=[False, False])
        res_df_full.to_csv(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_full.csv",
            index=False)
    else:
        res_df_full = pd.read_csv(
            "/lv_local/home/user/train_fb_ranker/output_results/baseline_model_choice_results_full.csv")

    process_top_model(res_df_full.model.iloc[0], test_paths, "test")
