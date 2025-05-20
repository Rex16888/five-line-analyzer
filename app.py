import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import io
import os

st.set_page_config(page_title="五線譜分析工具", page_icon="📈", layout="wide")

# ==== 自定義 CSS（深色主題 + 動畫 + 覆蓋） ====
st.markdown("""
<style>
html, body, .stApp {
    background-color: #111111 !important;
    color: white !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background-color: #1c1c1c;
    color: white;
    border-radius: 5px;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 0.5rem 1.5rem;
    border: none;
    border-radius: 5px;
    font-size: 16px;
    transition: 0.3s;
}
.stButton > button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
.logo {
    text-align: center;
    margin-bottom: 2rem;
    animation: fadeIn 1.5s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==== Logo 與標題 ====
st.markdown("""
<div class="logo">
    <h1>🎼 樂活五線譜分析工具</h1>
    <p>用最懶惰的方式，幫恁爸自動分析股票趨勢</p>
</div>
""", unsafe_allow_html=True)

# === 選單區塊 ===
st.sidebar.title("📚 功能選單")
st.sidebar.markdown("- 📥 匯入股票清單\n- 📈 斜率分析\n- 📊 五線譜圖表\n- 📋 下載報表")

# === 檔案上傳 ===
st.subheader("📤 匯入準備發財股票代碼清單")
uploaded_file = st.file_uploader("請上傳 CSV 或 Excel（每列一筆代碼）", type=["csv", "xlsx"])

# === 手動輸入 ===
st.subheader("✍️ 太閒手動輸入股票代碼")
manual_input = st.text_input("多筆代碼時用逗號分隔（如 AAPL, TSLA, MSFT）")

# === 區間選擇 ===
st.subheader("📆 選擇分析時間區間")
date_range = st.selectbox("選擇抓取歷史股價的區間：", ["1y", "2y", "3y", "5y", "10y"], index=3)

# === 合併代碼 ===
tickers = []
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_codes = pd.read_csv(uploaded_file)
        else:
            df_codes = pd.read_excel(uploaded_file)
        for col in df_codes.columns:
            if "代碼" in str(col):
                tickers += df_codes[col].dropna().astype(str).str.upper().tolist()
                break
        else:
            tickers += df_codes.iloc[:, 0].dropna().astype(str).str.upper().tolist()
    except Exception as e:
        st.error(f"❌ 檔案讀取失敗：{e}")

if manual_input:
    manual_list = [x.strip().upper() for x in manual_input.split(",") if x.strip()]
    tickers += manual_list

tickers = list(set(tickers))

# === 分析按鈕 ===
if st.button("🚀 開始分析") and tickers:
    results = []
    image_files = []
    st.markdown("---")
    st.subheader("📊 分析結果與圖表")

    progress = st.progress(0, text="資料載入中...")
    for i, ticker in enumerate(tickers):
        time.sleep(0.1)
        progress.progress((i + 1) / len(tickers), text=f"正在處理 {ticker}...")

        data = yf.download(ticker, period=date_range)
        if data.empty:
            st.warning(f"⚠ 無法取得 {ticker} 的資料")
            continue

        data = data.reset_index()
        data = data[['Date', 'Close']].dropna()
        data['Index'] = range(1, len(data) + 1)

        X = data['Index'].values.reshape(-1, 1)
        y = data['Close'].values

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        trend = model.predict(X)
        std_dev = np.std(y - trend)

        if slope > 0.5:
            trend_status = "📈 上升"
        elif slope < -0.5:
            trend_status = "📉 下跌"
        else:
            trend_status = "🔄 橫盤"

        results.append({
            '股票代碼': ticker,
            '斜率': slope,
            '趨勢判斷': trend_status,
            '截距': intercept,
            '標準差': std_dev
        })

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], y, label='股價', color='black')
        ax.plot(data['Date'], trend, label='T 趨勢線', linestyle='--')
        ax.plot(data['Date'], trend + std_dev, label='T+1S', linestyle=':')
        ax.plot(data['Date'], trend + 2 * std_dev, label='T+2S', linestyle=':')
        ax.plot(data['Date'], trend - std_dev, label='T-1S', linestyle=':')
        ax.plot(data['Date'], trend - 2 * std_dev, label='T-2S', linestyle=':')
        ax.set_title(f"{ticker} 的五線譜")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # 匯出為 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label=f"📥 下載 {ticker} 圖表 PNG",
            data=buf.getvalue(),
            file_name=f"{ticker}_五線譜.png",
            mime="image/png"
        )

    if results:
        df_results = pd.DataFrame(results)
        st.subheader("📋 分析報表 (斜率 + 趨勢判斷 + 標準差 + 截距)")
        st.dataframe(df_results, use_container_width=True)

        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 下載分析結果 CSV",
            data=csv,
            file_name="五線譜分析結果.csv",
            mime="text/csv"
        )
