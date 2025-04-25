import streamlit as st
import numpy as np
from scipy.stats import poisson

st.title("⚽ 手机版足球比分预测")
st.markdown("输入攻防参数，实时计算泊松分布概率！")

col1, col2 = st.columns(2)
with col1:
    home_attack = st.slider("主队进攻强度", 0.1, 3.0, 1.5)
    away_defense = st.slider("客队防守强度", 0.1, 3.0, 1.0)
with col2:
    away_attack = st.slider("客队进攻强度", 0.1, 3.0, 1.2)
    home_defense = st.slider("主队防守强度", 0.1, 3.0, 1.0)

home_xg = (home_attack + away_defense) / 2
away_xg = (away_attack + home_defense) / 2

max_goals = 6
home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]

score_matrix = np.outer(home_probs, away_probs)
top_scores = sorted(
    [(f"{i}-{j}", score_matrix[i][j]) for i in range(max_goals) for j in range(max_goals)],
    key=lambda x: -x[1]
)[:10]

st.subheader("📊 热门比分概率")
for score, prob in top_scores:
    st.write(f"{score}: {prob*100:.2f}%")