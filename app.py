from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import talib
import requests
import json
import google.generativeai as genai
import gc
import os
from itertools import dropwhile

app = Flask(__name__)

# Initialization
api_key = os.environ.get('API_KEY')
cm_url = os.environ.get('CM_URL')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

use_ollama = False
ollama_model = "deepseek-r1:8b"

BARS = 60




################################################################################################################################################################
def fetch_tw_whale(ticker):
  
  if ".TW" in ticker:

    # Get CMoney CK key first
    headers = {
      'Accept': 'application/json, text/javascript, */*; q=0.01',
      'Accept-Language': 'en-US,en;q=0.9',
      'Connection': 'keep-alive',
      'Referer': f'{cm_url}?action=mf&id=6446',
      'Sec-Fetch-Dest': 'empty',
      'Sec-Fetch-Mode': 'cors',
      'Sec-Fetch-Site': 'same-origin',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
      'X-Requested-With': 'XMLHttpRequest',
      'sec-ch-ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
    }
    
    params = {
      'action': 'mf',
      'count': str(BARS),
      'id': '2330',
      'ck': 'ML7EuNTM4B87LWAfCN94XIUVRMUVIHmjrEY^DHVQHBWwCRQrCXC3jMk$5OErz',
    }

    ck = ''
    r = requests.get(f'{cm_url}?action=mf&id=2330', headers=headers, verify=False)
    if r.status_code == 200:
      idx_b = r.text.index('var ck = "') + 10
      if idx_b > 0:
        idx_e = r.text.index('";', idx_b)
        ck = r.text[idx_b:idx_e]
        params['ck'] = ck
        print(f'CK: {ck}')
    

    if ck != '':
      params['id'] = ticker[:ticker.index('.')]
      r = requests.get(cm_url, params=params, headers=headers, verify=False)
      if r.status_code == 200:
        cm_data = r.json()
        records = [
          {
            "date": pd.to_datetime(c[0], unit='ms'),
            "mf": c[8]["MfOvrBuy"] if c[8] else 0,
            "mf_acc": c[8]["MfOvrBuySm"] if c[8] else 0,
            "b_s": c[8]["BuyerSm"] if c[8] else 0,
          }
          for c in cm_data.get("DataLine", [])
        ]

        df_mf = pd.DataFrame.from_records(records).set_index("date")

        df_mf_reset = df_mf.tail(BARS).reset_index()
        df_mf_reset['date'] = df_mf_reset['date'].dt.strftime('%Y-%m-%d')
        json_records = df_mf_reset.to_dict(orient='records')
        #print(json_records)
        del df_mf, df_mf_reset
        gc.collect()
        
        return json_records
    else:
      return {}
  else:
    return {}
    



################################################################################################################################################################
def fetch_stock_data(ticker):
  close = 'Close'
  MA_TYPE = 0
  
  stock = yf.Ticker(ticker)
  hist = stock.history(period="2y", auto_adjust=True)        
  hist['ATR'] = talib.ATR(hist['High'], hist['Low'], hist['Close'], timeperiod=5)
  
  hist.drop(['Open', 'High', 'Low', 'Stock Splits'], axis=1, inplace=True)
  gc.collect()

  hist['10MA'] = talib.SMA(hist[close], timeperiod=10)
  hist['20MA'] = talib.SMA(hist[close], timeperiod=20)
  hist['60MA'] = talib.SMA(hist[close], timeperiod=60)
  hist['200MA'] = talib.SMA(hist[close], timeperiod=200)  
  hist['Bollinger Band Upper'], hist['60MA'], hist['Bollinger Band Lower'] = talib.BBANDS(hist[close].values, timeperiod=60, nbdevup=2, nbdevdn=2, matype=MA_TYPE)    
  hist['RSI'] = talib.RSI(hist[close], timeperiod=14)
  hist['MACD'], hist['MACD Signal'], hist['MACD Hist'] = talib.MACD(hist[close], fastperiod=50, slowperiod=120, signalperiod=30)
  
  hist.drop(['MACD', 'MACD Signal'], axis=1, inplace=True)
  gc.collect()
  
  # Calculate the difference between closing price and 200MA
  hist['200MA Diff'] = (hist[close]-hist['200MA'])/hist['200MA']*100

  # Calculate the mean and standard deviation of the differences
  mean_diff = hist['200MA Diff'].mean()
  std_diff = hist['200MA Diff'].std()

  # Calculate the z-score
  hist['200MA Diff Z-Score'] = (hist['200MA Diff'] - mean_diff) / std_diff
  
  hist.drop(['200MA Diff'], axis=1, inplace=True)
  gc.collect()
  
  hist = hist.reset_index().tail(BARS)
  
  #print(hist)
  
  hist_dict = hist.to_dict(orient="records")
  
  financials = stock.financials.to_dict()
  quarterly_financials = stock.quarterly_financials.to_dict()
  cash_flow = stock.cash_flow.to_dict()
  quarterly_cashflow = stock.quarterly_cashflow.to_dict()
  #info = stock.info
  info = dict(dropwhile(lambda item: item[0] != 'previousClose', stock.info.items()))
  
  upgrades_downgrades = stock.upgrades_downgrades[:10].to_dict()
  eps_trend = stock.eps_trend.to_dict()
  revenue_estimate = stock.revenue_estimate.to_dict()
  
  options = stock.options[:8]
  options_data = {}
  for date in options:
    opt = stock.option_chain(date)
    options_data[date] = {
        "calls": opt.calls.to_dict(orient="records"),
        "puts": opt.puts.to_dict(orient="records")
    }

  gc.collect()
  
  return {
    "history": hist_dict,
    "financials": financials,
    "quarterly_financials": quarterly_financials,
    "cash_flow": cash_flow,
    "quarterly_cashflow": quarterly_cashflow,
    "info": info,
    "upgrades_downgrades": upgrades_downgrades,
    "eps_trend": eps_trend,
    "revenue_estimate": revenue_estimate,
    "options": options_data
  }




################################################################################################################################################################
def ollama_generate(prompt, model='llama3'):
  print(f"Use Ollama: {model}")
 
  url = 'http://localhost:11434/api/generate'
  payload = {
    "model": model,
    "prompt": prompt,
    "stream": False
  }
  try:
    response = requests.post(url, json=payload, timeout=2400)
    response.raise_for_status()
    data = response.json()
    return data.get('response', '')
  except Exception as e:
    return f"Ollama 呼叫失敗: {e}"




################################################################################################################################################################
@app.route('/analysis/', methods=['GET', 'POST'])
def gemini_analysis():
  gc.collect()
  
  analysis = None
  error = None
  ticker = ''
  model_name = 'gemini-2.5-pro'  # 預設值

  if request.method == 'POST':
    ticker = request.form.get('ticker', '').strip()
    additional_prompt = request.form.get('additional_prompt', '').strip()
    model_name = request.form.get('model', 'gemini-2.0-flash')
    
    if not ticker:
      error = "請輸入股票代碼"
    else:
      try:
        stock_data = fetch_stock_data(ticker)
        prompt_prefix = f'請根據{ticker}的歷史股價與技術分析（含10MA, 20MA, 60MA, 200MA, RSI, ATR, Volume, MACD Histogram, Bollinger Band, 200MA Diff Z-Score, Dividends）配合對應的成交量 (Volume)，財報 (financials, quarterly_financials, cash_flow, quarterly_cashflow, info)，與期權資料，{additional_prompt}，列出近期財報亮點與分析師評論 (upgrades_downgrades, eps_trend, revenue_estimate) 的整理，且產生一份繁體中文個股分析報告，首先列出目前價格與關鍵支持價位，然後內容包含基本面 (數字要有YoY加減速的分析，以及free cashflow的研究，並且根據年度財報預估與當季累積財報數字，預估後面一兩季的營收獲利起伏)、技術面 (配合成交量分析, 例如是否有價量背離) 與期權市場的觀察與建議。 若資料中有台灣股市 (mainforce_tw) 主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)，順便分析主力吃貨或出貨狀況。'
        prompt = f'{prompt_prefix}\n資料如下：\n{stock_data}'
        #print('----------------------------------------')
        #print(prompt)
        #print('----------------------------------------')
        if use_ollama == True:
          analysis = ollama_generate(prompt, model=ollama_model)
        else:
          model = genai.GenerativeModel(model_name)
          response = model.generate_content(prompt)
          analysis = response.text
      except Exception as e:
        error = f"分析過程發生錯誤: {e}"

  return render_template('analysis.html', analysis=analysis, error=error, ticker=ticker, model=model_name)




################################################################################################################################################################
################################################################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)
