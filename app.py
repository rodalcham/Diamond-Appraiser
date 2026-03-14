import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#  Page config 
st.set_page_config(page_title="Diamond Appraiser", page_icon="💎", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e0e;
    color: #f0ece4;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #e8d5a3;
    letter-spacing: 0.02em;
    margin-bottom: 0;
}
.subtitle {
    font-size: 0.95rem;
    color: #888;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.price-box {
    background: linear-gradient(135deg, #1a1a1a, #222);
    border: 1px solid #e8d5a3;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.price-label {
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.price-value {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #e8d5a3;
    margin: 0.3rem 0;
}
.divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 2.5rem 0;
}
</style>
""", unsafe_allow_html=True)

#  Load & prepare data 
@st.cache_data
def load_data():
    df = sns.load_dataset('diamonds')
    df = df[['carat', 'cut', 'color', 'clarity', 'price']]
    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['carat'] > 0]
    df['cut']     = pd.Categorical(df['cut'],     categories=['Fair','Good','Very Good','Premium','Ideal'], ordered=True)
    df['color']   = pd.Categorical(df['color'],   categories=['J','I','H','G','F','E','D'], ordered=True)
    df['clarity'] = pd.Categorical(df['clarity'], categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], ordered=True)
    return df

@st.cache_data
def train_model(df):
    df_model = df.copy()
    df_model['cut']     = df_model['cut'].cat.codes
    df_model['color']   = df_model['color'].cat.codes
    df_model['clarity'] = df_model['clarity'].cat.codes

    X = df_model[['carat', 'cut', 'color', 'clarity']].values
    y = df_model['price'].values
    X = np.c_[np.ones(X.shape[0]), X]

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    y_pred = X_test @ theta
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    return theta, r2, rmse

df = load_data()
theta, r2, rmse = train_model(df)

#  Header 
st.markdown('<p class="title">💎 Diamond Appraiser</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate the market value of a diamond using a linear regression model</p>', unsafe_allow_html=True)

#  Appraiser Form 
st.markdown("### Appraise a Diamond")

col1, col2 = st.columns(2)

with col1:
    carat   = st.number_input("Carat", min_value=0.1, max_value=5.0, value=0.5, step=0.01)
    cut     = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

with col2:
    color   = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Encode inputs
cut_code     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'].index(cut)
color_code   = ['J', 'I', 'H', 'G', 'F', 'E', 'D'].index(color)
clarity_code = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(clarity)

x_input = np.array([[1, carat, cut_code, color_code, clarity_code]])
predicted_price = float((x_input @ theta).item())
predicted_price = max(0, predicted_price)

st.markdown(f"""
<div class="price-box">
    <div class="price-label">Estimated Price</div>
    <div class="price-value">${predicted_price:,.0f}</div>
    <div class="price-label">Model R² {r2:.4f} &nbsp;|&nbsp; RMSE ${rmse:,.0f}</div>
</div>
""", unsafe_allow_html=True)

#  Divider 
st.markdown('<hr class="divider">', unsafe_allow_html=True)

#  Histogram 
st.markdown("### Price Distribution")

filter_by = st.selectbox("Filter histogram by", ['None', 'cut', 'color', 'clarity'])

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#e0e0e0")
ax.set_facecolor("#EAEAEA")

if filter_by == 'None':
    ax.hist(df['price'], bins=50, color='#e8d5a3', edgecolor='#0e0e0e', linewidth=0.4)
else:
    categories = df[filter_by].cat.categories
    colors = plt.cm.YlOrBr(np.linspace(0.2, 0.9, len(categories)))
    for cat, col in zip(categories, colors):
        subset = df[df[filter_by] == cat]['price']
        ax.hist(subset, bins=50, alpha=0.6, label=str(cat), color=col, edgecolor='none')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#f0ece4', fontsize=8)

ax.set_xlabel('Price (USD)', color='#888', fontsize=10)
ax.set_ylabel('Count', color='#888', fontsize=10)
ax.tick_params(colors='#666')
for spine in ax.spines.values():
    spine.set_edgecolor('#2a2a2a')

st.pyplot(fig)