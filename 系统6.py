import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os

# ==================== 设置 matplotlib 字体为 STSong ====================
plt.rcParams['font.sans-serif'] = ['STSong']   # 宋体，支持 μ 和 ₂
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="尘肺病分期预测系统", layout="wide")
st.title("矿山环境矿工职业健康评价系统")
st.markdown("### 输入矿工的职业暴露信息，预测尘肺病分期（0期、I期、II期、III期）")

DATA_FILE = "粉尘数据.xlsx"
MODEL_FILE = "stacking_model.pkl"
SCALER_FILE = "scaler.pkl"
LE_FILE = "le.pkl"

# 训练使用的原始列名（内部，包含尖括号）
RAW_COLUMNS = ['工序', '确诊年龄', '接尘时间', '粉尘浓度',
               '<2um', '2-5um', '5-10um', '>10um', '游离SiO2', '尘肺病阶段']
FEATURE_COLS = RAW_COLUMNS[:-1]

# 用于显示的特征名称（美观，带 μ 和 下标）
DISPLAY_COLS = ['工序', '确诊年龄', '接尘时间', '粉尘浓度',
                '<2μm', '2-5μm', '5-10μm', '>10μm', '游离SiO₂']

@st.cache_resource
def train_and_save_model():
    df_raw = pd.read_excel(DATA_FILE, sheet_name="Sheet1", header=None, skiprows=2)
    df_raw.columns = RAW_COLUMNS
    for col in df_raw.columns[1:]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw.dropna(inplace=True)

    X = df_raw[FEATURE_COLS].copy()
    y = df_raw['尘肺病阶段'] - 1

    le = LabelEncoder()
    X['工序'] = le.fit_transform(X['工序'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  objective='multi:softmax', num_class=4,
                                  random_state=42, eval_metric='mlogloss')
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('lr', lr)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, stack_method='auto'
    )
    stacking_clf.fit(X_train, y_train)

    joblib.dump(stacking_clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, LE_FILE)
    return stacking_clf, scaler, le

if not os.path.exists(MODEL_FILE):
    with st.spinner("首次运行，正在训练模型，请稍候..."):
        model, scaler, le = train_and_save_model()
    st.success("模型训练完成！")
else:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    le = joblib.load(LE_FILE)

# ==================== 输入表单 ====================
col1, col2 = st.columns(2)

with col1:
    process = st.selectbox("工序", ["落煤", "割煤", "打眼", "放炮", "运转", "检修", "多工序"])
    age = st.number_input("确诊年龄 (岁)", min_value=20, max_value=80, value=45, step=1)
    exposure = st.number_input("接尘时间 (年)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
    dust = st.number_input("粉尘浓度 (mg/m³)", min_value=0.0, max_value=300.0, value=30.0, step=1.0)
    sio2 = st.number_input("游离SiO₂含量 (%)", min_value=0.0, max_value=25.0, value=10.0, step=0.5)

with col2:
    d_lt2 = st.number_input("<2μm 分散度 (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    d_2_5 = st.number_input("2-5μm 分散度 (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    d_5_10 = st.number_input("5-10μm 分散度 (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    d_gt10 = st.number_input(">10μm 分散度 (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

# 工序编码映射
process_map = {"落煤":0, "割煤":1, "打眼":2, "放炮":3, "运转":4, "检修":5, "多工序":6}
process_enc = process_map[process]

# 构建输入 DataFrame（列名必须与训练时一致，使用尖括号）
input_df = pd.DataFrame([[process_enc, age, exposure, dust,
                          d_lt2, d_2_5, d_5_10, d_gt10, sio2]],
                        columns=FEATURE_COLS)
features_scaled = scaler.transform(input_df)

if st.button("开始预测", type="primary"):
    proba = model.predict_proba(features_scaled)[0]
    pred_class = np.argmax(proba)
    stages = ["0期", "I期", "II期", "III期"]
    pred_stage = stages[pred_class]
    pred_prob = proba[pred_class] * 100

    st.markdown("---")
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.metric("预测分期", pred_stage, delta=f"概率 {pred_prob:.1f}%")
    with col_res2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
        bars = ax.bar(stages, proba, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("概率")
        ax.set_title("各尘肺病分期预测概率")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f"{p:.2f}", ha='center', fontsize=10)
        st.pyplot(fig)

    # 特征重要性图
    st.subheader("特征对预测的贡献度")
    rf_model = model.named_estimators_['rf']
    importance = rf_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [DISPLAY_COLS[i] for i in sorted_idx]
    sorted_vals = importance[sorted_idx]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars2 = ax2.barh(sorted_names, sorted_vals, color='#2C5F8A')
    ax2.set_xlabel("重要性 (基尼不纯度下降)")
    ax2.set_title("特征重要性 (基于随机森林)")
    max_val = max(sorted_vals) * 1.15
    ax2.set_xlim(0, max_val)
    for i, (bar, val) in enumerate(zip(bars2, sorted_vals)):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center', fontsize=9)
    st.pyplot(fig2)

    # ==================== 修正后的输入特征值表格 ====================
    st.markdown("### 输入特征值")
    # 收集当前输入的实际数值（顺序与 DISPLAY_COLS 一致）
    input_values = [process, age, exposure, dust, d_lt2, d_2_5, d_5_10, d_gt10, sio2]
    # 创建 DataFrame，两列：特征（显示名）和当前值（数值）
    feat_df = pd.DataFrame({
        "特征": DISPLAY_COLS,
        "当前值": input_values
    })
    # 使用 st.table 或 st.dataframe，确保所有值正常显示
    st.table(feat_df)

    # 健康建议
    st.markdown("### 健康管理建议")
    if pred_stage == "0期":
        st.info("目前未发现尘肺病征象，建议每年常规职业健康体检，继续做好粉尘防护。")
    elif pred_stage == "I期":
        st.warning("提示：初步诊断为 I 期尘肺病，建议缩短体检周期至半年一次，加强个体防护，必要时考虑调岗。")
    elif pred_stage == "II期":
        st.error("提示：初步诊断为 II 期尘肺病，应尽快调离粉尘岗位，接受规范化治疗和康复训练。")
    else:
        st.error("提示：初步诊断为 III 期尘肺病，立即脱离粉尘环境，住院治疗并申请工伤保险。")