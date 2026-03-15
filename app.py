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
def train_model(df): # We train Both models now! 
    df_model = df.copy()
    df_model['cut']     = df_model['cut'].cat.codes # We convert categoricals into numbers 
    df_model['color']   = df_model['color'].cat.codes
    df_model['clarity'] = df_model['clarity'].cat.codes

    X = df_model[['carat', 'cut', 'color', 'clarity']].values # We define inputs
    
    X = np.c_[X, X[:, 0]**2]  # add carat squared as a 5th feature, I've heard this might help the model to learn that the relation carat-price is not linear. Carat^2 - Price is closer to lineal
    X_means = X.mean(axis=0) # We want to normalize the input to have mean=0 and std=1, so we calculate mean
    X_stds  = X.std(axis=0) # And std
    X = (X - X_means) / X_stds # Then we normalize, this should help the model be more reliable in extreme scenarios

    y = df_model['price'].values # Defining Output

    X = np.c_[np.ones(X.shape[0]), X] # Adding a Bias Column
    
    np.random.seed(17012002) # Introducing a random seed for reproducibility
    indices = np.random.permutation(len(X)) # We shuffle the data (originally it was ordered)
    X, y = X[indices], y[indices]

    split = int(0.9 * len(X)) # We split into train / test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train #Linear Equation to get Thetas 
    
    # Lets train a Log-linear model as well, just for fun
    log_y_train = np.log(y_train) # We change the values int log space
    theta_log = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ log_y_train


    y_pred = X_test @ theta # We test!
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

    y_pred_log = X_test @ theta_log  # The log-Linear Model as well
    exp_y_pred_log = np.exp(y_pred_log)  # We return the values to dollar space
    ss_res_log = np.sum((np.log(y_test) - y_pred_log) ** 2) # Converting lo log space to get an accurate comaprison
    ss_tot_log = np.sum((np.log(y_test) - np.mean(np.log(y_test))) ** 2) # Same
    
    r2 = 1 - (ss_res / ss_tot) # Here we get the r^2 and rmse
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2_log  = 1 - ss_res_log / ss_tot_log
    rmse_log = np.sqrt(np.mean((y_test - exp_y_pred_log) ** 2))
    
    return theta, r2, rmse, theta_log, r2_log, rmse_log, y_pred, exp_y_pred_log, y_test, X_means, X_stds

df = load_data()
theta, r2, rmse, theta_log, r2_log, rmse_log, y_pred, exp_y_pred_log, y_test, X_means, X_stds = train_model(df)

#  Header 
st.markdown('<p style="font-family: \'Playfair Display\', serif; font-size: 3.5rem; color: #e8d5a3; letter-spacing: 0.02em; margin-bottom: 0;">💎 Diamond Appraiser</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate the market value of a diamond using a linear regression model</p>', unsafe_allow_html=True)

#  Appraiser Form 
st.markdown("### Appraise a Diamond")

max_train_carat = float(df['carat'].quantile(0.95))

col1, col2 = st.columns(2)

with col1:
    carat   = st.number_input("Carat", min_value=0.1, max_value=5.1, value=1.0, step=0.1)

    cut     = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

with col2:
    color   = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    
model = st.selectbox("Model", ['Linear Regression', 'Log-Linear Regression'])
if carat > max_train_carat:
    st.warning(f"⚠️ Carat value exceeds 95% of training data (max: {max_train_carat:.1f}). Predictions will be unreliable.")

# Encode inputs and run them throught the selected model
cut_code     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'].index(cut)
color_code   = ['J', 'I', 'H', 'G', 'F', 'E', 'D'].index(color)
clarity_code = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'].index(clarity)

x_raw = np.array([carat, cut_code, color_code, clarity_code, carat**2])# Normalizing again
x_norm = (x_raw - X_means) / X_stds
x_input = np.array([[1, *x_norm]])

if model == 'Linear Regression': # Checking what model we used to display the right data
    predicted_price = max(0, float((x_input @ theta).item()))
    r2_display, rmse_display = r2, rmse
else:
    predicted_price = float(np.exp(x_input @ theta_log).item())
    r2_display, rmse_display = r2_log, rmse_log

st.markdown(f"""
<div class="price-box">
    <div class="price-label">Estimated Price ({model})</div>
    <div class="price-value">${predicted_price:,.0f}</div>
    <div class="price-label">Model R² {r2_display:.4f} &nbsp;|&nbsp; RMSE ${rmse_display:,.0f}</div>
</div>
""", unsafe_allow_html=True)

#  Divider 
st.markdown('<hr class="divider">', unsafe_allow_html=True)

#  Histogram 
st.markdown("### Price Distribution")

filter_by = st.selectbox("Filter histogram by", ['None', 'cut', 'color', 'clarity'])

CATEGORY_COLORS = { # Picking some nice colors
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


# Let's compare Linear regression vs Log-Linear Regression by plotting predicted vs real prices

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("### Predicted vs Actual Prices")
st.caption("Sampled from the test set. A perfect model would follow the diagonal line.")

sample_n = 500
rng = np.random.default_rng(42)
idx = rng.choice(len(y_test), size=min(sample_n, len(y_test)), replace=False)

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('#ffffff')

for ax, y_pred_plot, title, r2_val, rmse_val in zip(
    axes,
    [y_pred, exp_y_pred_log],
    ["Linear Regression", "Log-Linear Regression"],
    [r2, r2_log],
    [rmse, rmse_log]
):
    ax.set_facecolor('#f5f5f5')
    ax.scatter(y_test[idx], y_pred_plot[idx], alpha=0.3, s=8, color='#4a90d9')

    # Perfect prediction line
    max_val = min(y_test.max(), y_pred.max(), exp_y_pred_log.max())
    ax.plot([0, max_val], [0, max_val], color='#c9a84c', linewidth=1.5, linestyle='--', label='Perfect fit')

    # Highlight selected model with gold border
    is_selected = (model == 'Linear Regression' and title == 'Linear Regression') or \
                  (model == 'Log-Linear Regression' and title == 'Log-Linear Regression')
    for spine in ax.spines.values():
        spine.set_edgecolor('#c9a84c' if is_selected else '#cccccc')
        spine.set_linewidth(2 if is_selected else 1)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_title(f"{title}\nR² = {r2_val:.4f}  |  RMSE ${rmse_val:,.0f}", color='#333333', fontsize=11)
    ax.set_xlabel('Actual Price (USD)', color='#333333', fontsize=9)
    ax.set_ylabel('Predicted Price (USD)', color='#333333', fontsize=9)
    ax.tick_params(colors='#333333')
    ax.legend(facecolor='#ffffff', edgecolor='#cccccc', labelcolor='#333333', fontsize=8)

plt.tight_layout()
st.pyplot(fig2)