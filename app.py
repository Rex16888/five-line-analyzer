import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import io

st.set_page_config(page_title="五線譜分析工具", layout="wide")
st.title("📈 樂活五線譜自動分析工具")

# === 匯入檔案 ===
uploaded_file = st.file_uploader("📤 請上傳股票代碼清單（CSV 或 Excel）", type=["csv", "xlsx"])

# === 手動輸入股票代碼 ===
manual_input = st.text_input("✍️ 手動輸入股票代碼（可多支，用逗號分隔，如 AAPL, TSLA）")

# === 選擇資料區間 ===
date_range = st.selectbox("選擇資料區間：", ["1y", "2y", "3y", "5y", "10y"], index=3)

# === 整理股票代碼 ===
tickers = []

# 檔案匯入（含欄位名稱辨識）
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_codes = pd.read_csv(uploaded_file)
        else:
            df_codes = pd.read_excel(uploaded_file)

        found = False
        for col in df_codes.columns:
            if "代碼" in str(col):
                tickers += df_codes[col].dropna().astype(str).str.upper().tolist()
                found = True
                break
        if not found:
            tickers += df_codes.iloc[:, 0].dropna().astype(str).str.upper().tolist()
    except Exception as e:
        st.error(f"❌ 檔案讀取失敗：{e}")

# 手動輸入
if manual_input:
    manual_list = [x.strip().upper() for x in manual_input.split(",") if x.strip()]
    tickers += manual_list

# 去除重複
tickers = list(set(tickers))

if st.button("開始分析") and tickers:
    results = []

    for ticker in tickers:
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

        results.append({
            '股票代碼': ticker,
            '斜率': slope,
            '截距': intercept,
            '標準差': std_dev
        })

        # 畫圖
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

    # 顯示分析結果表格
    if results:
        df_results = pd.DataFrame(results)
        st.subheader("分析報表")
        st.dataframe(df_results, use_container_width=True)

        # 提供 CSV 下載
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 下載分析結果 CSV",
            data=csv,
            file_name="五線譜分析結果.csv",
            mime="text/csv"
        )

