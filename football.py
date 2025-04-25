import streamlit as st
import numpy as np
from scipy.stats import poisson

# 页面标题
st.title("⚽ 足球比分预测（泊松分布模型）")
st.markdown("通过泊松分布计算两队进球概率，预测比分分布。")

# 输入参数
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("主队名称", "曼城")
    home_attack = st.slider("主队进攻强度（场均进球）", 0.1, 3.0, 1.8)
    away_defense = st.slider("客队防守强度（场均失球）", 0.1, 3.0, 1.2)
with col2:
    away_team = st.text_input("客队名称", "利物浦")
    away_attack = st.slider("客队进攻强度（场均进球）", 0.1, 3.0, 1.5)
    home_defense = st.slider("主队防守强度（场均失球）", 0.1, 3.0, 1.0)

# 计算预期进球（xG）
home_xg = (home_attack + away_defense) / 2
away_xg = (away_attack + home_defense) / 2

# 泊松分布计算概率
max_goals = 10  # 最大模拟进球数
home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]

# 生成比分概率矩阵
score_matrix = np.outer(home_probs, away_probs)

# 显示结果
st.subheader(f"预期进球（xG）: {home_team}: {home_xg:.2f} | {away_team}: {away_xg:.2f}")

# 热门比分概率
st.subheader("🔍 热门比分概率")
top_scores = sorted(
    [(f"{i}-{j}", score_matrix[i][j]) for i in range(max_goals) for j in range(max_goals)],
    key=lambda x: -x[1]
)[:10]  # 取前10大概率

for score, prob in top_scores:
    st.write(f"{score}: {prob*100:.2f}%")

# 可视化概率分布
st.subheader("📊 比分概率热力图")
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    score_matrix[:6, :6],  # 显示0-5球的矩阵
    annot=True,
    fmt=".2%",
    xticklabels=range(6),
    yticklabels=range(6),
    cmap="YlOrRd",
    ax=ax
)
ax.set_xlabel(away_team + " 进球数")
ax.set_ylabel(home_team + " 进球数")
st.pyplot(fig)

# 与庄家赔率对比（可选）
st.subheader("🎯 价值投注分析")
bookie_odds = st.number_input(f"输入庄家对 {home_team} 胜的赔率（如2.10）", min_value=1.1, max_value=20.0, value=2.10)
implied_prob = 1 / bookie_odds
model_prob = np.sum(np.triu(score_matrix, k=1))  # 主队胜概率（上三角矩阵和）
st.write(
    f"- 庄家隐含胜率: {implied_prob*100:.2f}%\n"
    f"- 模型预测胜率: {model_prob*100:.2f}%\n"
)
if model_prob > implied_prob:
    st.success("✅ 模型认为存在价值投注机会！")
else:
    st.warning("❌ 无显著价值。")