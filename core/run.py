# !/usr/bin/env python3
# -*- coding:utf-8 _*-

import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from xgboost import plot_importance

from imblearn.over_sampling import SMOTE

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score






def process_data(df):
    cate_cols = [
        "PROVINCE",
        "SEX",
        "AGE",
        "occupation",
        "mobile_fixed_days",
        "adr_stability_days",
        "consume_steady_byxs_1y",
        "use_mobile_2_cnt_1y",
        "activity_area_stability",
        "last_1y_total_active_biz_cnt",
        "have_car_flag",
        "have_fang_flag",
        "last_6m_avg_asset_total",
        "last_3m_avg_asset_total",
        "last_1m_avg_asset_total",
        "last_1y_avg_asset_total",
        "tot_pay_amt_6m",
        "tot_pay_amt_3m",
        "tot_pay_amt_1m",
        "ebill_pay_amt_6m",
        "ebill_pay_amt_3m",
        "ebill_pay_amt_1m",
        "avg_puc_sdm_last_1y",
        "xfdc_index",
        "pre_1y_pay_cnt",
        "pre_1y_pay_amount",
        "auth_fin_last_1m_cnt",
        "auth_fin_last_3m_cnt",
        "auth_fin_last_6m_cnt",
        "credit_pay_amt_1m",
        "credit_pay_amt_3m",
        "credit_pay_amt_6m",
        "credit_pay_amt_1y",
        "credit_pay_months_1y",
        "credit_total_pay_months",
        "credit_duration",
        "positive_biz_cnt_1y",
        "ovd_order_cnt_1m",
        "ovd_order_amt_1m",
        "ovd_order_cnt_3m",
        "ovd_order_amt_3m",
        "ovd_order_cnt_6m",
        "ovd_order_amt_6m",
        "ovd_order_cnt_12m",
        "ovd_order_amt_12m",
        "ovd_order_c0t_3m_m1_status",
        "ovd_order_c0t_6m_m1_status",
        "ovd_order_c0t_12m_m1_status",
        "ovd_order_c0t_12m_m3_status",
        "ovd_order_c0t_12m_m6_status",
        "ovd_order_c0t_21_m3_status",
        "ovd_order_c0t_21_m6_status",
        "ovd_order_c0t_51_m3_status",
        "ovd_order_c0t_51_m6_status",
        "relevant_stability",
        "sns_pii",
    ]

    # 处理类别标签 one-hot
    df = pd.get_dummies(df, columns=cate_cols)
    return df


def train_by_xgb():
    data = pd.read_csv("data/data.csv")
    df = process_data(df=data)
    # 提取X和Y标签
    cols = [col for col in df.columns if col not in ["id_rank", "CREDIT_FLAG"]]
    X = df[cols]
    y = df["CREDIT_FLAG"]
    print(X.shape)

    print("training xgb model...")
    # 处理类别不均衡问题
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_resample(X.values, y.values)
    # 3:1 将数据换分出训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res)
    clf = XGBClassifier(
        learning_rate=0.1,
        n_estimators=240,
        silent=True,
        objective="binary:logistic",
        booster="gbtree",
        eval_metric="logloss",
        max_depth=6,
    )
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=True,
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(clf, height=0.5, ax=ax, max_num_features=100)
    plt.show()

    # 在测试集验证效果
    print("evaluation on test data:")
    y_pred = clf.predict(x_test)
    label = pd.DataFrame()
    label["y_pred"] = y_pred
    y_prob = clf.predict_proba(x_test)[:, 1]
    print("Evaluation on test precision ：%s" % accuracy_score(y_test, y_pred))
    print("Evaluation on test auc score  ：%s" % roc_auc_score(y_test, y_prob))

    # 根据分类概率得出用户评分
    print("forming the xgb final result")
    all_data = pd.read_csv("data/data.csv")
    df = process_data(df=all_data)
    # 提取X和Y标签
    cols = [col for col in df.columns if col not in ["id_rank", "CREDIT_FLAG"]]
    X = df[cols]
    y = df["CREDIT_FLAG"]
    all_data["label"] = clf.predict_proba(X.values)[:, 0]
    all_data["label"] = all_data["label"].apply(lambda x: round(x * 95, 2))
    # all_data[['USER_ID', 'label']].to_csv('xgb_submission.csv', index=None)
    all_data[["id_rank", "label"]].to_csv("results/xgb_submission.csv", index=None)

    result = all_data[["id_rank", "label"]]
    return result


def train_by_kmeans():
    data = pd.read_csv("data/data.csv")
    data = process_data(df=data)
    print("training k-means")

    clf = KMeans(
        copy_x=True,
        init="k-means++",
        max_iter=400,
        n_clusters=10,
        n_init=10,
        random_state=42,
        tol=0.0001,
        verbose=True,
    )
    # 去除USER_ID一列
    cols = [col for col in data.columns if col not in ["id_rank"]]
    clf.fit(data[cols])
    # clf.fit(data)

    data["label"] = clf.labels_  # 对原数据表进行类别标记
    # 按照人数从低到高排序
    label_score = dict()
    as_labels = data["label"].value_counts(ascending=True).index.tolist()
    print(as_labels)
    for index, label in enumerate(as_labels):
        label_score[label] = (index + 1) * 9

    data["label"].value_counts().plot(kind="barh")
    plt.show()

    data["label"] = data["label"].apply(lambda x: label_score[x])
    # data[['USER_ID', 'label']].to_csv('submission.csv', index=None)

    result = data[["id_rank", "label"]]
    return result


def main():
    """
    根据xgb分类模型以及kmeans聚类结果，降权求和
    :return:
    """
    weight_kmean = 0.5
    weight_xgb = 0.5

    result_xgb = train_by_xgb()
    result_kmeans = train_by_kmeans()
    result_xgb.drop_duplicates(subset=["id_rank"], keep="first", inplace=True)
    result_kmeans.drop_duplicates(subset=["id_rank"], keep="first", inplace=True)

    print(len(result_xgb))
    print(len(result_kmeans))

    result = pd.merge(result_xgb, result_kmeans, how="inner", on="id_rank")
    result["label"] = 100 - (
        result["label_x"] * weight_xgb + result["label_y"] * weight_kmean
    )
    # result.to_csv('result.csv')

    result["id_rank"] = result["id_rank"].astype("int32")
    result["label"] = result["label"].apply(lambda x: round(x, 2))
    result.rename(columns={"id_rank": "用户ID", "label": "信用评分"}, inplace=True)
    result[["用户ID", "信用评分"]].to_csv("results/Credit_Predict.csv", index=False)


if __name__ == "__main__":
    main()
