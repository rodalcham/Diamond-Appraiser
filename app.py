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
    font-size: 3.5rem;
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
    background: linear-gradient(135deg, #f5f0e8, #fffdf7);
    border: 1px solid #c9a84c;
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
    color: #b8860b;
    margin: 0.3rem 0;
}
.divider {
    border: none;
    border-top: 1px solid #cccccc;
    margin: 2.5rem 0;
}
</style>
""", unsafe_allow_html=True)

#  Load & prepare data 
@st.cache_data # Makes the function run only once
def load_data():
    df = sns.load_dataset('diamonds') #We take the Data from seaborn because the link didn't work
    df = df[['carat', 'cut', 'color', 'clarity', 'price']]
    df = df.dropna() # We get rid of empty or impossible values
    df = df[df['price'] > 0]
    df = df[df['carat'] > 0] 
    df['cut']     = pd.Categorical(df['cut'],     categories=['Fair','Good','Very Good','Premium','Ideal'], ordered=True) # We order the categories, to improve results
    df['color']   = pd.Categorical(df['color'],   categories=['J','I','H','G','F','E','D'], ordered=True)
    df['clarity'] = pd.Categorical(df['clarity'], categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], ordered=True)
    return df

@st.cache_data #Makes the function run only once
def train_model(df):
    df_model = df.copy()
    df_model['cut']     = df_model['cut'].cat.codes # We convert categoricals into numbers 
    df_model['color']   = df_model['color'].cat.codes
    df_model['clarity'] = df_model['clarity'].cat.codes

    X = df_model[['carat', 'cut', 'color', 'clarity']].values # We define inputs and output
    y = df_model['price'].values
    X = np.c_[np.ones(X.shape[0]), X] # Adding a Bias Column

    indices = np.random.permutation(len(X)) # We shuffle the data (originally it was ordered)
    X, y = X[indices], y[indices]

    split = int(0.9 * len(X)) # We split into train / test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train #Linear Equation to get Thetas 

    y_pred = X_test @ theta # We test!
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    return theta, r2, rmse

df = load_data()
theta, r2, rmse = train_model(df)

#  Header 
st.markdown('<p style="font-family: \'Playfair Display\', serif; font-size: 3.5rem; color: #e8d5a3; letter-spacing: 0.02em; margin-bottom: 0;">💎 Diamond Appraiser</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate the market value of a diamond using a linear regression model</p>', unsafe_allow_html=True)

#  Appraiser Form 
st.markdown("### Appraise a Diamond")

col1, col2 = st.columns(2)

with col1:
    carat   = st.number_input("Carat", min_value=0.1, max_value=5.1, value=1.0, step=0.1)
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

CATEGORY_COLORS = {
    'cut':     ['#e63946', "#e4904c", '#e9c46a', "#1d832f", "#3FCB49"],
    'color':   ['#e63946', '#e4904c', '#e9c46a', '#1d832f', '#3FCB49', "#53f4e9", "#3ea9fb"],
    'clarity': ['#e63946', '#e4904c', '#e9c46a', '#1d832f', '#3FCB49', "#53f4e9", '#3ea9fb', "#002af9"],
}

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#f5f5f5')

if filter_by == 'None':
    ax.hist(df['price'], bins=50, color='#4a90d9', edgecolor='white', linewidth=0.5)
else:
    categories = df[filter_by].cat.categories
    colors = CATEGORY_COLORS[filter_by]
    subsets = [df[df[filter_by] == cat]['price'].values for cat in categories]
    ax.hist(subsets, bins=50, stacked=True, label=[str(c) for c in categories], color=colors, edgecolor='white', linewidth=0.4)
    ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333', fontsize=8)

ax.set_xlabel('Price (USD)', color='#333333', fontsize=10)
ax.set_ylabel('Count', color='#333333', fontsize=10)
ax.tick_params(colors='#333333')
for spine in ax.spines.values():
    spine.set_edgecolor('#cccccc')

st.pyplot(fig)