from flask import Flask, request, render_template, Response, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import talib
import requests
import json
import gc
import os
from itertools import dropwhile
from io import StringIO
import re

import requests

from urllib.parse import urljoin
import random

from pyecharts.charts import Bar, Tab
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode

from datetime import date, datetime, timedelta
from curl_cffi import requests
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line

import time
import plotly
import plotly.express as px




# Initialization
api_key = os.environ.get('API_KEY')
cm_url = os.environ.get('CM_URL')
cm_url2 = os.environ.get('CM_URL2')
si_url = os.environ.get('SI_URL')
tw_sf_url = os.environ.get('TW_SF_URL')


use_ollama = False
ollama_model = "deepseek-r1:8b"

BARS = 200


app = Flask(__name__)




################################################################################################################################################################
@app.route('/link/')
def link():
  links = []
  for rule in app.url_map.iter_rules():
    if "GET" in rule.methods and not rule.rule.startswith('/static'):
        links.append((rule.endpoint, rule.rule))
  html = '''
  <!DOCTYPE html>
  <html lang="zh">
  <head>
    <meta charset="UTF-8">
    <title>所有路由</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <div class="container mt-5">
        <h1 class="mb-4">Links</h1>
        <ul class="list-group">
        {% for endpoint, url in links %}
            <li class="list-group-item">
                <a href="{{ url }}" class="link-primary">{{ url }}</a>
                <span class="badge bg-secondary ms-2">{{ endpoint }}</span>
            </li>
        {% endfor %}
        </ul>
    </div>
  </body>
  </html>
  '''
  return render_template_string(html, links=links)




################################################################################################################################################################
################################################################################################################################################################
def fetch_tw_whale(ticker):
  
  return_value = {}
  
  if ".TW" in ticker:

    # Get CMoney CK key first
    headers = {
      'Accept': 'application/json, text/javascript, */*; q=0.01',
      'Accept-Language': 'en-US,en;q=0.9',
      'Connection': 'keep-alive',
      'Referer': f'{cm_url}?action=mf&id={ticker}',
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
      r = requests.get(cm_url2, params=params, headers=headers, verify=False)
      if r.status_code == 200:
        cm_data = r.json()
        if cm_data != None:
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
          return_value = df_mf_reset.to_dict(orient='records')

  return return_value




################################################################################################################################################################
def fetch_short_stats(ticker):
  
  return_value = {}
  
  if ".TW" not in ticker:
    headers_si = {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Language': 'zh-TW,zh-CN;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6',
      'Cache-Control': 'max-age=0',
      'Connection': 'keep-alive',
      'Referer': 'https://www.google.com/',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'same-origin',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
    }
      
    r = requests.get(f'{si_url}/quote/{ticker}/short-interest', headers=headers_si, verify=False)
    if r.status_code == 200:
      html = r.text
      idx_b = html.find('"shortInterest":[{')
      if idx_b > -1:
        idx_e = html.find('],', idx_b)
        if idx_e > -1:
          json_txt = '[' + html[idx_b+17:idx_e+1]
          df = pd.read_json(StringIO(json_txt), orient='records')
          df['recordDate'] = pd.to_datetime(df['recordDate']).dt.strftime('%Y-%m-%d')
          if 'shortPercentOfFloat' in df.columns:
            rename_cols = {'recordDate': 'date', 'daysToCover': 'SR', 'shortPercentOfFloat': 'SF'}
          else:
            rename_cols = {'recordDate': 'date', 'daysToCover': 'SR'}
            
          df = df.rename(columns=rename_cols)[rename_cols.values()].tail(20)

          return_value = df.to_dict(orient='records')

  return return_value




################################################################################################################################################################
def fetch_tw_financing_stats(ticker):
  
  return_value = {}
  
  if ".TW" in ticker:

    url = f"{tw_sf_url}?no={ticker[:ticker.index('.')]}&m=mg"

    r = requests.get(url, timeout=10, verify=False)
    
    if r.status_code == 200:
      r.encoding = 'utf-8'
      html = r.text

      col_list = ["'融資餘額(張)'", "'融券餘額(張)'", "'借券賣出餘額(張)'"]
      data_list = []

      for c in col_list:
        idx_b = html.find(f"{c},\r\n") + len(c) + 1
        if idx_b > (len(c) + 21):
          idx_e = html.find(',\r\n', idx_b)
          data = html[idx_b:idx_e].strip()
          json_data = json.loads(data[6:])    # Remove 'data: ' and string to json list
          data_list.append(json_data)

      dfs = [pd.DataFrame(l, columns=['date', col_list[i]]).set_index('date') for i, l in enumerate(data_list)]
      if len(dfs) == 3:
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, unit='ms').strftime('%Y-%m-%d')
        df.rename(columns={"'融資餘額(張)'": "BB", "'融券餘額(張)'": "SB", "'借券賣出餘額(張)'": "LSB"}, inplace=True)
  
        df_reset = df.tail(BARS).reset_index()
        return_value = df_reset.to_dict(orient='records')

  return return_value




################################################################################################################################################################
def simplified_options(stock):
  result = []
  for date in stock.options:
    opt = stock.option_chain(date)
    for opt_type, label in zip(['calls', 'puts'], ['call', 'put']):
      df = getattr(opt, opt_type)
      df = df.copy()
      df['premium_est'] = df['lastPrice'] * df['volume'] * 100
      filtered = df[df['premium_est'] > 200_000]
      for _, row in filtered.iterrows():
        result.append({
          'date': date,
          'type': label,
          'strike': round(row['strike'], 4),
          'lastPrice': round(row['lastPrice'], 4),
          'volume': round(row['volume'], 4),
          'openInterest': round(row['openInterest'], 4),
          'premiumEstimated': round(row['premium_est'], 4),
          'inTheMoney': bool(row['inTheMoney'])
        })
  return result




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
  #hist['60MA'] = talib.SMA(hist[close], timeperiod=60)
  hist['200MA'] = talib.SMA(hist[close], timeperiod=200)  
  hist['BBU'], hist['60MA'], hist['BBD'] = talib.BBANDS(hist[close].values, timeperiod=60, nbdevup=2, nbdevdn=2, matype=MA_TYPE)    
  hist['RSI'] = talib.RSI(hist[close], timeperiod=14)
  hist['MACD'], hist['MACD Signal'], hist['MACDH'] = talib.MACD(hist[close], fastperiod=50, slowperiod=120, signalperiod=30)
  
  hist.drop(['MACD', 'MACD Signal'], axis=1, inplace=True)
  gc.collect()
  
  # Calculate the difference between closing price and 200MA
  hist['200MA Diff'] = (hist[close]-hist['200MA'])/hist['200MA']*100

  # Calculate the mean and standard deviation of the differences
  mean_diff = hist['200MA Diff'].mean()
  std_diff = hist['200MA Diff'].std()

  # Calculate the z-score
  hist['200MADZ'] = (hist['200MA Diff'] - mean_diff) / std_diff
  
  hist.drop(['200MA Diff'], axis=1, inplace=True)
  gc.collect()
  
  hist = hist.round(2)
  hist = hist.reset_index().tail(BARS)
  hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
  
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
  
  """
  options = stock.options[:8]
  options_data = {}
  for date in options:
    opt = stock.option_chain(date)
    options_data[date] = {
      "calls": opt.calls.round(4).to_dict(orient="records"),
      "puts": opt.puts.round(4).to_dict(orient="records")
    }
  """
  options_data = simplified_options(stock)
  
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
    "options": options_data,
    "mainforce_tw": fetch_tw_whale(ticker),
    "short_stats": fetch_short_stats(ticker),
    "securities_financing_tw": fetch_tw_financing_stats(ticker)
  }




################################################################################################################################################################
def ollama_generate(prompt, model='llama3'):
  print(f"Use Ollama: {model}")
  
  headers = {
    "Content-Type": "application/json"
  }

  data = {
    "model": model,  # 請確認這個模型已經在本地 ollama 中存在
    "messages": [
      {
        "role": "user",
        "content": prompt
      }
    ],
    "stream": False  # 若設為 True，會變成 stream 回傳
  }
 
  url = "http://localhost:11434/api/chat"
  response = requests.post(url, headers=headers, data=json.dumps(data), timeout=600)
  
  if response.status_code == 200:
    result = response.json()
    return(result["message"]["content"])
  else:
    print(f"❌ Ollama error：{response.status_code}")
    print(response.text)




################################################################################################################################################################
def gemini_generate_content(prompt, model_name, api_key):
  url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
  headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key
  }
  data = {
    "contents": [
       {
         "parts": [
           {"text": prompt}
         ]
       }
    ],
    "tools": [
       {
         "google_search": {}
       }
    ]
  }
  
  response = requests.post(url, headers=headers, json=data, timeout=600)
  if response.status_code == 200:
    result = response.json()
    # 取出回應內容
    #return result['candidates'][0]['content']['parts'][0]['text']
    
    # ★ 修正 2：整合所有 parts 的文字
    try:
      if 'candidates' in result and result['candidates']:
          content = result['candidates'][0].get('content', {})
          parts = content.get('parts', [])
          
          # 使用 join 將所有 part 的 text 串接起來
          full_text = "".join([part.get('text', '') for part in parts])
          
          return full_text
      else:
          return "No candidates returned."
            
    except (KeyError, IndexError) as e:
      return f"Error parsing response: {e}"
  
  else:
    raise Exception(f"❌ Gemini API error: {response.status_code} {response.text}")





################################################################################################################################################################
def dict_to_table(data: list[dict], limit=200, csv=True) -> str:
  if not data:
    return "無資料"
  df = pd.DataFrame(data).tail(limit)
  if csv == True:
    return df.to_csv(index=False)
  else:
    return df.to_string(index=False)




################################################################################################################################################################
def dict_to_table_finance(data: list[dict], csv=True) -> str:
  if not data:
    return "無資料"
  df = pd.DataFrame(data)
  
  if csv == True:
    return df.to_csv(index=True)
  else:
    return df.to_string(index=True)




################################################################################################################################################################
@app.route('/aia/', methods=['GET', 'POST'])
def gemini_analysis():
  gc.collect()
  
  analysis = None
  error = None
  ticker = ''
  model_name = 'gemini-2.5-pro'  # 預設值

  if request.method == 'POST':
    ticker = request.form.get('ticker', '').strip()
    additional_prompt = request.form.get('additional_prompt', '').strip()
    model_name = request.form.get('model', 'gemini-2.5-pro')
    
    if not ticker:
      error = "請輸入股票代碼"
    else:
      try:
        stock_data = fetch_stock_data(ticker)
        """
        prompt_prefix = f'請根據{ticker}的歷史股價與技術分析（含10MA， 20MA， 60MA， 200MA， RSI， ATR， Volume， MACD Histogram (MACDH)， 60MA Bollinger Band (BBU， BBD)， 200MA Diff Z-Score (200MADZ)）配合對應的成交量 (Volume)，財報 (financials， quarterly_financials， cash_flow， quarterly_cashflow， info)，與期權資料，{additional_prompt}，列出近期財報亮點與分析師評論 (upgrades_downgrades， eps_trend， revenue_estimate) 的整理，且產生一份繁體中文個股分析報告，首先列出公司近期業務，然後列出目前價格與關鍵支持價位，以及根據財報預測數據所推算的未來股價，然後內容包含基本面 (數字要有YoY加減速的分析，以及free cashflow的研究，並且根據年度財報預估與當季累積財報數字，預估後面一兩季的營收獲利起伏與對應的PE PS PB ratio，並且以表格列出每季EPS與營收增減的速度與加速度)、技術面 (配合成交量分析， 例如是否有價量背離或技術指標與股價背離) 與期權市場的觀察與建議。 若資料中有台灣股市 (mainforce_tw) 主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)，順便分析主力吃或出貨狀況。若資料中有short_stats，根據SF (short floating) 與SR (short ratio) 分析市場空單狀況及嘎空可能性。若資料中有台灣股市 (securities_financing_tw) 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)，分析市場空單狀況，嘎空可能性以及未來主力操作方向。'
        prompt = f'{prompt_prefix}\n資料如下：\n{stock_data}'
        #print('----------------------------------------')
        #print(prompt)
        #print('----------------------------------------')
        """
        history_table = dict_to_table(stock_data['history'])
        mf_tw_table = dict_to_table(stock_data['mainforce_tw'])
        short_table = dict_to_table(stock_data['short_stats'])
        sf_tw_table = dict_to_table(stock_data['securities_financing_tw'])
        financials_table = dict_to_table_finance(stock_data['financials'])
        financials_q_table = dict_to_table_finance(stock_data['quarterly_financials'])
        cashflow_table = dict_to_table_finance(stock_data['cash_flow'])
        cashflow_q_table = dict_to_table_finance(stock_data['quarterly_cashflow'])
        #info_table = dict_to_table_finance(stock_data['info'])
        updown_table = dict_to_table_finance(stock_data['upgrades_downgrades'])
        eps_trend_table = dict_to_table_finance(stock_data['eps_trend'])
        revenue_estimate_table = dict_to_table_finance(stock_data['revenue_estimate'])
        option_table = dict_to_table_finance(stock_data['options'])

        gc.collect()
       
        print('----------------------------------------')
        prompt = f"""
你是一個資深的華爾街分析師，擅長從基本面技術面籌碼面產生股票分析報告。請根據 {ticker} 的下列資料進行分析並產生一份繁體中文個股分析報告，首先從網路上搜尋近半年公司相關新聞並以表格方式總結其對股價與經營業務的影響，然後列出目前價格與關鍵支持價位，以及根據財報預測數據所推算的未來股價，然後內容包含基本面 (數字要有YoY加減速的分析，以及自由現金流的研究，並且根據年度財報預估與當季累積財報數字，預估後面一兩季的營收獲利起伏與對應的PE/PS/PB ratio，並且以表格列出每季EPS與營收增減的速度與加速度)、技術面 (配合成交量分析，例如是否有價量背離或技術指標與股價背離，或是型態上有破底翻、假突破、杯柄型態、上升旗型、下降旗型等典型股價走勢型態) 與期權市場的觀察與建議。 若mf_tw_table資料中有台灣股市主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)，順便分析主力吃或出貨狀況。若資料中short_table有值，根據SF (short floating) 與SR (short ratio) 分析市場空單狀況及嘎空可能性。若資料中有台灣股市 (sf_tw_table) 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)，分析市場空單狀況，嘎空可能性以及未來主力操作方向。 其他需求: {additional_prompt}

[技術面]
csv table 欄位縮寫: MACD Histogram (MACDH)， 60MA Bollinger Band (BBU， BBD)， 200MA Diff Z-Score (200MADZ)
{history_table}

[台股主力籌碼]
csv table 欄位縮寫: 主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)
{mf_tw_table}

[台股融資融券與借券賣出餘額]
csv table 欄位縮寫: 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)
{sf_tw_table}

[空單狀況]
csv table 欄位縮寫: SF (short floating)， SR (short ratio)
{short_table}

[財報資料]
###
公司概況:
{stock_data['info']}

###
年度財報: csv table 
{financials_table}

###
季度財報: csv table 
{financials_q_table}

###
年度現金流: csv table 
{cashflow_table}

###
季度現金流: csv table 
{cashflow_q_table}

###
EPS年度趨勢: csv table 
{eps_trend_table}

###
營收預估: csv table 
{revenue_estimate_table}

###
評等變化: csv table 
{updown_table}


[期權市場]
csv table
{option_table}
        """
        print(prompt)
        print('----------------------------------------')

        if use_ollama == True:
          analysis = ollama_generate(prompt, model=ollama_model)
        else:
          #import google.generativeai as genai
          #genai.configure(api_key=api_key)
          #model = genai.GenerativeModel(model_name)
          #response = model.generate_content(prompt)
          #analysis = response.text
          analysis = gemini_generate_content(prompt, model_name, api_key)
      except Exception as e:
        error = f"分析過程發生錯誤: {e}"

  return render_template('analysis.html', analysis=analysis, error=error, ticker=ticker, model=model_name)




################################################################################################################################################################
@app.route('/suityourself/', methods=['GET', 'POST'])
def gemini_analysis_user():
    gc.collect()
    analysis = None
    error = None
    ticker = ''
    model_name = 'gemini-2.5-pro'

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        additional_prompt = request.form.get('additional_prompt', '').strip()
        model_name = request.form.get('model', 'gemini-2.5-pro')
        gemini_key = request.form.get('gemini_key', '').strip()

        if not ticker:
            error = "請輸入股票代碼"
        elif not gemini_key:
            error = "請輸入有效的 Gemini API Key"
        else:
            try:
                # 假設 fetch_stock_data 和 gemini_generate_content 已定義
                stock_data = fetch_stock_data(ticker)
                history_table = dict_to_table(stock_data['history'])
                mf_tw_table = dict_to_table(stock_data['mainforce_tw'])
                short_table = dict_to_table(stock_data['short_stats'])
                sf_tw_table = dict_to_table(stock_data['securities_financing_tw'])
                financials_table = dict_to_table_finance(stock_data['financials'])
                financials_q_table = dict_to_table_finance(stock_data['quarterly_financials'])
                cashflow_table = dict_to_table_finance(stock_data['cash_flow'])
                cashflow_q_table = dict_to_table_finance(stock_data['quarterly_cashflow'])
                #info_table = dict_to_table_finance(stock_data['info'])
                updown_table = dict_to_table_finance(stock_data['upgrades_downgrades'])
                eps_trend_table = dict_to_table_finance(stock_data['eps_trend'])
                revenue_estimate_table = dict_to_table_finance(stock_data['revenue_estimate'])
                option_table = dict_to_table_finance(stock_data['options'])

                gc.collect()
               
                print('----------------------------------------')
                prompt = f"""
你是一個資深的華爾街分析師，擅長從基本面技術面籌碼面產生股票分析報告。請根據 {ticker} 的下列資料進行分析並產生一份繁體中文個股分析報告，首先從網路上搜尋近半年公司相關新聞並以表格方式總結其對股價與經營業務的影響，然後列出目前價格與關鍵支持價位，以及根據財報預測數據所推算的未來股價，然後內容包含基本面 (數字要有YoY加減速的分析，以及自由現金流的研究，並且根據年度財報預估與當季累積財報數字，預估後面一兩季的營收獲利起伏與對應的PE/PS/PB ratio，並且以表格列出每季EPS與營收增減的速度與加速度)、技術面 (配合成交量分析，例如是否有價量背離或技術指標與股價背離，或是型態上有破底翻、假突破、杯柄型態、上升旗型、下降旗型等典型股價走勢型態) 與期權市場的觀察與建議。 若mf_tw_table資料中有台灣股市主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)，順便分析主力吃或出貨狀況。若資料中short_table有值，根據SF (short floating) 與SR (short ratio) 分析市場空單狀況及嘎空可能性。若資料中有台灣股市 (sf_tw_table) 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)，分析市場空單狀況，嘎空可能性以及未來主力操作方向。 其他需求: {additional_prompt}

[技術面]
csv table 欄位縮寫: MACD Histogram (MACDH)， 60MA Bollinger Band (BBU， BBD)， 200MA Diff Z-Score (200MADZ)
{history_table}

[台股主力籌碼]
csv table 欄位縮寫: 主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)
{mf_tw_table}

[台股融資融券與借券賣出餘額]
csv table 欄位縮寫: 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)
{sf_tw_table}

[空單狀況]
csv table 欄位縮寫: SF (short floating)， SR (short ratio)
{short_table}

[財報資料]
###
公司概況:
{stock_data['info']}

###
年度財報: csv table 
{financials_table}

###
季度財報: csv table 
{financials_q_table}

###
年度現金流: csv table 
{cashflow_table}

###
季度現金流: csv table 
{cashflow_q_table}

###
EPS年度趨勢: csv table 
{eps_trend_table}

###
營收預估: csv table 
{revenue_estimate_table}

###
評等變化: csv table 
{updown_table}


[期權市場]
csv table
{option_table}
"""
                print(prompt)
                print('----------------------------------------')
                analysis = gemini_generate_content(prompt, model_name, gemini_key)
            except Exception as e:
                error = f"分析過程發生錯誤: {e}"

    return render_template('analysis_user.html', analysis=analysis, error=error, ticker=ticker, model=model_name)




################################################################################################################################################################
################################################################################################################################################################
def rewrite_html(html, base_url):
  # 將 src/href 內的絕對或相對路徑改寫成 proxy 路徑
  def repl(match):
    orig_url = match.group(2)
    # 處理相對路徑
    if not orig_url.startswith('http'):
      from urllib.parse import urljoin
      orig_url = urljoin(base_url, orig_url)
    return f'{match.group(1)}/proxy?url={orig_url}{match.group(3)}'

  # 只處理 src 和 href
  pattern = r'((?:src|href)=["\'])([^"\']+)(["\'])'
  return re.sub(pattern, repl, html, flags=re.IGNORECASE)




################################################################################################################################################################
@app.route('/proxy')
def proxy():
  target_url = request.args.get('url')
  if not target_url:
    return "請提供 ?url= 參數", 400

  try:
    resp = requests.get(target_url, headers={
      'User-Agent': request.headers.get('User-Agent', 'Mozilla/5.0')
    }, timeout=10)
    content_type = resp.headers.get('Content-Type', '')

    if 'text/html' in content_type:
      # 只重寫 HTML
      html = resp.text
      html = rewrite_html(html, target_url)
      return Response(html, status=resp.status_code, content_type=content_type)
    else:
      # 其他資源直接回傳
      return Response(resp.content, status=resp.status_code, content_type=content_type)

  except Exception as e:
      return f"Error: {e}", 500




################################################################################################################################################################
################################################################################################################################################################
no_list = list(range(1, 29660))
random.shuffle(no_list)
no_index = 0




################################################################################################################################################################
@app.route('/hokkien/')
def hokkien():
  return render_template('hokkien.html')




################################################################################################################################################################
def replace_button_with_audio(html):
  # 用 re 找出 button 的 data-src
  def repl(m):
    audio_url = m.group(1)
    # 你可以自訂 audio 樣式，這裡用 controls 會有原生播放icon
    return f'''
    <audio controls style="vertical-align: middle; height: 20px width: 20px;">
        <source src="{audio_url}" type="audio/mpeg">
        您的瀏覽器不支援音訊播放。
    </audio>
      '''
  # 把 button 換成 audio
  new_html = re.sub(
    r'<button[^>]*data-src="([^"]+)"[^>]*>.*?</button>',
    repl,
    html,
    flags=re.S
  )
  return new_html




################################################################################################################################################################
@app.route('/api/random')
def hokkien_random_word():
  global no_index, no_list
  max_retry = 10
  base_url = 'https://sutian.moe.edu.tw/'
  
  for _ in range(max_retry):
    # 取下一個不重複的 no
    if no_index >= len(no_list):
      random.shuffle(no_list)
      no_index = 0
    no = no_list[no_index]
    no_index += 1

    url = f'https://sutian.moe.edu.tw/zh-hant/su/{no}/'
    try:
      resp = requests.get(url,  timeout=5, verify=False)
    except Exception:
      continue
   
    if resp.status_code != 200:
      continue

    html = resp.text

    # 找到 div.row.justify-content-center
    match = re.search(r'<div class="row justify-content-center".*?</div>\s*</div>', html, re.S)
    if not match:
      continue
    div_html = match.group(0)

    # 補上 <a> 的完整網址
    #div_html = re.sub(
    #  r'href="(?!http)([^"]+)"',
    #  lambda m: f'href="{urljoin(base_url, m.group(1).lstrip("/"))}"',
    #  div_html
    #)
    
    div_html = re.sub(
      r'href="(?!http)([^"]+)"',
      lambda m: (
        f'href="{m.group(1)}"' if m.group(1).startswith('#')
        else f'href="{urljoin(base_url, m.group(1).lstrip("/"))}"'
      ),
      div_html
    )

    # 補上 <img> 的完整網址
    div_html = re.sub(
      r'src="(?!http)([^"]+)"',
      lambda m: f'src="{urljoin(base_url, m.group(1).lstrip("/"))}"',
      div_html
    )

    # 找出 button 的 data-src
    button_match = re.search(r'<button[^>]*data-src="([^"]+)"', div_html)
    audio_url = urljoin(base_url, button_match.group(1)) if button_match else ''

    # 補上 <button> 的完整 data-src
    div_html = re.sub(
      r'data-src="(?!http)([^"]+)"',
      lambda m: f'data-src="{urljoin(base_url, m.group(1).lstrip("/"))}"',
      div_html
    )

    div_html = replace_button_with_audio(div_html)

    return jsonify({'no': no, 'html': div_html, 'audio_url': audio_url})

  # 如果10次都沒找到
  return jsonify({'no': None, 'html': '<div>查無資料</div>', 'audio_url': ''})




################################################################################################################################################################
################################################################################################################################################################
def generate_option_tabs(ticker: str):
  stock = yf.Ticker(ticker)
  expirations = stock.options

  tab = Tab()
  all_options_list = []

  for expiry in expirations:
    try:
      opt_chain = stock.option_chain(expiry)
      calls = opt_chain.calls
      puts = opt_chain.puts
    except Exception:
      continue

    # 計算 premium
    calls["premium_est"] = calls["lastPrice"] * calls["volume"] * 100
    puts["premium_est"] = puts["lastPrice"] * puts["volume"] * 100

    options = pd.concat([calls.assign(type="Call"), puts.assign(type="Put")])
    options["expiry"] = expiry

    # 過濾條件
    options = options[(options["volume"] > 0) & (options["premium_est"] > 200_000)]
    if options.empty:
        continue

    options["premium_K"] = options["premium_est"] / 1000
    options = options.sort_values(by="premium_K", ascending=False)

    all_options_list.append(options)

    # 繪圖
    contracts = options["contractSymbol"].tolist()
    premiums = options["premium_K"].round(1).tolist()
    color_list = ["#FF4C4C" if t=="Call" else "#2ECC71" for t in options["type"]]

    bar = (
      Bar(init_opts=opts.InitOpts(width="1280px", height="720px"))
      .add_xaxis(contracts)
      .add_yaxis("Premium (K USD)", premiums,
                 #itemstyle_opts=opts.ItemStyleOpts(color="auto"),
                 itemstyle_opts=opts.ItemStyleOpts(color=JsCode("""
                    function(params) {
                      var colors = %s;
                      return colors[params.dataIndex];
                    }
                    """ % color_list)
                 ),
                 label_opts=opts.LabelOpts(is_show=True, position="top"))
      .set_global_opts(
        title_opts=opts.TitleOpts(title=f"{ticker.upper()} Options (Expiry {expiry})"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, font_size=8)),
        yaxis_opts=opts.AxisOpts(name="Premium (K USD)"),
      )
    )

    tab.add(bar, expiry)

  # 總覽 tab
  if all_options_list:
    all_options = pd.concat(all_options_list)
    all_top10 = all_options.sort_values(by="premium_K", ascending=False).head(10)

    contracts = (all_top10["contractSymbol"] + " (" + all_top10["expiry"] + ")").tolist()
    premiums = all_top10["premium_K"].round(1).tolist()
    color_list = ["#FF4C4C" if t=="Call" else "#2ECC71" for t in all_top10["type"]]

    overview_bar = (
      Bar(init_opts=opts.InitOpts(width="1280px", height="720px"))
      .add_xaxis(contracts)
      .add_yaxis("Premium (K USD)", premiums,
                 #itemstyle_opts=opts.ItemStyleOpts(color="auto"),
                 itemstyle_opts=opts.ItemStyleOpts(color=JsCode("""
                    function(params) {
                      var colors = %s;
                      return colors[params.dataIndex];
                    }
                    """ % color_list)
                 ),
                 label_opts=opts.LabelOpts(is_show=True, position="top"))
      .set_global_opts(
        title_opts=opts.TitleOpts(title=f"{ticker.upper()} Options Overview (Top 10 Premium)"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, font_size=8)),
        yaxis_opts=opts.AxisOpts(name="Premium (K USD)"),
      )
    )
    
    tab.add(overview_bar, "Overview")

  return tab.render_embed()




################################################################################################################################################################
@app.route("/optionpremium/", methods=["GET", "POST"])
def optionpremium():
  chart_html = None
  ticker = None
  if request.method == "POST":
    ticker = request.form.get("ticker")
    if ticker:
      chart_html = generate_option_tabs(ticker)

  return render_template("optionpremium.html", chart_html=chart_html, ticker=ticker)




################################################################################################################################################################
################################################################################################################################################################
def get_stock_data(ticker, start_date, end_date, session, crumb="F7GXvns0Eji"):

  start_epoch = int(datetime.combine(start_date, datetime.min.time()).timestamp())
  end_epoch = int(datetime.combine(end_date, datetime.min.time()).timestamp())
  url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=history&includeAdjustedClose=true&events=div%2Csplits&crumb={crumb}"
  
  headers = {'user-agent': 'Mozilla/5.0'}
  r = session.get(url, headers=headers, timeout=5)
  r.raise_for_status()
  data = r.json()
  result = data["chart"]["result"][0]
  quote = result["indicators"]["quote"][0]
  adjclose = result["indicators"]["adjclose"][0]["adjclose"]
  df = pd.DataFrame({
    "Date": pd.to_datetime(result["timestamp"], unit='s'),
    "Open": quote["open"],
    "High": quote["high"],
    "Low": quote["low"],
    "Close": quote["close"],
    "Adj Close": adjclose,
    "Volume": quote["volume"]
  }).set_index("Date")
  df.name = ticker
  return df




################################################################################################################################################################
def calculate_variation(df):
  
  df['Adj Close Var'] = (df['Adj Close'] / df['Adj Close'].iloc[0]) * 100
  return df




################################################################################################################################################################
def align_dataframes(dfs):
  
  min_len = min(len(df) for df in dfs)
  base_index = min(range(len(dfs)), key=lambda i: len(dfs[i]))
  base_dates = dfs[base_index].index
  for i, df in enumerate(dfs):
    if len(df) != min_len:
      dfs[i] = df.reindex(base_dates, method='ffill')
      dfs[i].name = df.name
  return dfs, base_index




################################################################################################################################################################
def compute_beta(df1, df2):
  m = df1['Adj Close'].pct_change().dropna()
  t = df2['Adj Close'].pct_change().dropna()
  min_len = min(len(m), len(t))
  m, t = m[-min_len:], t[-min_len:]
  cov = np.cov(m, t)[0][1]
  var = np.var(m)
  return cov / var if var != 0 else np.nan




################################################################################################################################################################
@app.route('/compare', methods=['GET', 'POST'])
def compare():
  
  if request.method == 'POST':
    tickers = request.form.get('tickers')
    days = int(request.form.get('days', 1800))
  else:
    tickers = request.args.get('tickers')
    days = int(request.args.get('days', 1800))
    
  if not tickers:
    return "Please provide tickers parameter, e.g. ?tickers=AAPL,MSFT", 400
  tickers = [t.strip() for t in tickers.replace(' ', ',').split(',') if t.strip()]
  if len(tickers) < 1:
    return "Please provide at least one ticker.", 400

  today = date.today()
  start_date = today - timedelta(days=days)
  session = requests.Session(impersonate="chrome")
 
  stock_dfs = []
  errors = []
  for ticker in tickers:
    try:
      df = get_stock_data(ticker, start_date, today, session)
      df = calculate_variation(df)
      stock_dfs.append(df)
    except Exception as e:
      errors.append(f"{ticker}: {e}")

  if not stock_dfs:
    return "No data fetched.<br>" + "<br>".join(errors), 500

  stock_dfs, base_idx = align_dataframes(stock_dfs)

  # Calculate beta
  beta_dict = {}
  base_df = stock_dfs[0]
  for i in range(1, len(stock_dfs)):
    beta_value = compute_beta(base_df, stock_dfs[i])
    beta_dict[stock_dfs[i].name] = beta_value

  # Beta string
  beta_str = "\n".join([f"{name} / {base_df.name}: β={beta_value:.2f}" for name, beta_value in beta_dict.items()])

  # Stats string
  stats_string = ""
  for df in stock_dfs:
    stats_string += f'{df.name}: δ={df["Adj Close Var"].iloc[-1] - df["Adj Close Var"].iloc[0]:5.2f}%, σ={df["Adj Close Var"].std():5.2f}%\n'
  
  # Plot
  line = Line(init_opts=opts.InitOpts(page_title=" vs ".join(tickers), height='900px', width='1880px'))
  dates = stock_dfs[base_idx].index.strftime('%Y%m%d').tolist()
  line.add_xaxis(xaxis_data=dates)
  for df in stock_dfs:
    line.add_yaxis(
      series_name=df.name,
      y_axis=df["Adj Close Var"].map('{:.2f}'.format).tolist(),
      is_smooth=False,
      is_symbol_show=False,
      is_hover_animation=False,
      linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.9)
    )
  
  line.set_global_opts(
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10)),
    yaxis_opts=opts.AxisOpts(is_scale=False, splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.5))),
    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross", textstyle_opts=opts.TextStyleOpts(font_size=12)),
    legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(font_size=12)),
    datazoom_opts=[
      opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0], range_start=0, range_end=100, is_realtime=False),
      opts.DataZoomOpts(is_show=True, xaxis_index=[0], type_="slider", pos_top="98%", range_start=0, range_end=100, is_realtime=False),
    ],
    title_opts=opts.TitleOpts(
      title=stats_string,
      subtitle=beta_str,  # beta in subtitle
      pos_left='10%',
      pos_top='10%',
      title_textstyle_opts=opts.TextStyleOpts(font_size=12),
      subtitle_textstyle_opts=opts.TextStyleOpts(font_size=12)
    ),
    toolbox_opts=opts.ToolboxOpts(is_show=True, feature={"dataZoom": {"yAxisIndex": "none"}, "restore": {}, "saveAsImage": {}}),
  )

  # Return HTML
  return line.render_embed()




################################################################################################################################################################
@app.route('/performance/', methods=['GET'])
def performance_diff():
  return '''
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <title>Stock Compare</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
          body {
              background: #f8f9fa;
          }
          .container {
              max-width: 600px;
              margin-top: 80px;
              background: #fff;
              border-radius: 12px;
              box-shadow: 0 2px 16px rgba(0,0,0,0.08);
              padding: 32px 32px 24px 32px;
          }
          .form-label {
              font-weight: 500;
          }
          .btn-primary {
              width: 100%;
              font-size: 1.1rem;
              padding: 10px;
          }
          h2 {
              text-align: center;
              margin-bottom: 32px;
              font-weight: 700;
              color: #2c3e50;
          }
      </style>
  </head>
  <body>
      <div class="container">
          <h2>Stock Performance Comparison</h2>
          <form action="/compare" method="post">
              <div class="mb-3">
                  <label for="tickers" class="form-label">Tickers (comma separated):</label>
                  <input type="text" class="form-control" id="tickers" name="tickers" value="^GSPC,AAPL" required>
              </div>
              <div class="mb-3">
                  <label for="days" class="form-label">Days:</label>
                  <input type="number" class="form-control" id="days" name="days" value="1800" min="1" required>
              </div>
              <button type="submit" class="btn btn-primary">Compare</button>
          </form>
          <div class="text-center mt-4" style="color:#888;font-size:0.95em;">
              Example: <code>^GSPC,AAPL,MSFT,GOOG</code> &nbsp; | &nbsp; Days: <code>3650</code><br>
              Following tickers can be put as 1st position for beta calculation.<br>
              <code>^GSPC=S&P 500, ^IXIC=NASDAQ, ^DJI=Dow Jones, ^TWII=TAIEX</code>
              
          </div>
      </div>
  </body>
  </html>
  '''




################################################################################################################################################################
################################################################################################################################################################
def get_expirations(ticker):
  stock = yf.Ticker(ticker)
  return stock.options




################################################################################################################################################################
def get_option_chain(ticker, expiration):
  stock = yf.Ticker(ticker)
  chain = stock.option_chain(expiration)

  del stock
  gc.collect()

  return chain.calls, chain.puts




################################################################################################################################################################
def calculate_max_pain(calls, puts):
  all_strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
  pain = {}

  for strike in all_strikes:
    total_loss = 0
    for _, row in calls.iterrows():
      if row['strike'] < strike:
        loss = row['openInterest'] * (strike - row['strike'])
        total_loss += loss
    for _, row in puts.iterrows():
      if row['strike'] > strike:
        loss = row['openInterest'] * (row['strike'] - strike)
        total_loss += loss
    pain[strike] = total_loss

  del all_strikes
  gc.collect()

  return min(pain, key=pain.get)




################################################################################################################################################################
def build_chart_option(calls, puts, ticker, max_pain, underlying_price):

  df_calls = calls[['strike', 'openInterest']].dropna()
  df_puts = puts[['strike', 'openInterest']].dropna()

  # 計算選擇權賣方的總損失
  strikes = sorted(set(df_calls['strike']).union(set(df_puts['strike'])))
  strike_labels = [str(s) for s in strikes]

  call_losses = []
  put_losses = []
  for expiry_price in strikes:
    # 看漲選擇權損失：價內 (strike < expiry_price)
    call_loss = df_calls[df_calls['strike'] < expiry_price].apply(
        lambda r: (expiry_price - r['strike']) * r['openInterest'], axis=1).sum()
    # 看跌選擇權損失：價內 (strike > expiry_price)
    put_loss = df_puts[df_puts['strike'] > expiry_price].apply(
        lambda r: (r['strike'] - expiry_price) * r['openInterest'], axis=1).sum()
    call_losses.append(float(call_loss))
    put_losses.append(float(put_loss))

  """
  mark_line = {
      "symbol": ["none", "none"],
      "label": {"formatter": "{b}: {c}", "position": "insideMiddle"},
      "lineStyle": {"type": "dashed"},
      "data": [
          {"xAxis": str(max_pain), "name": "Max Pain", "lineStyle": {"color": "blue"}},
          {"xAxis": str(round(underlying_price, 2)), "name": "Underlying", "lineStyle": {"color": "orange"}}
      ]
  }
  """
  mark_line = {
    "symbol": ["none", "none"],
    "label": {"formatter": "{b}: {c}", "position": "insideMiddle"},
    "lineStyle": {"type": "dashed"},
    "data": []
  }

  # 保證轉成字串
  if max_pain is not None:
    mark_line["data"].append({
      "xAxis": str(max_pain),
      "name": "Max Pain",
      "lineStyle": {"color": "blue"}
    })

  if underlying_price is not None:
    # 找到離 underlying_price 最近的 strike（讓 x 軸可以對得上）
    closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
    mark_line["data"].append({
      "xAxis": str(closest_strike),
      "name": "Underlying",
      "lineStyle": {"color": "orange"}
    })

  # chart1：選擇權賣方的總損失
  chart1 = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["Call Loss", "Put Loss"]},
    "xAxis": {
        "type": "category",
        "data": strike_labels,
        "name": "履約價",
        "axisLabel": {"rotate": 45}
    },
    "yAxis": {
        "type": "value",
        "name": "Total Loss ($)",
        "min": "dataMin",
        "max": "dataMax"
    },
    "series": [
        {"name": "Call Loss", "type": "bar", "data": call_losses, "itemStyle": {"color": "#d62728"}, "markLine": mark_line},
        {"name": "Put Loss", "type": "bar", "data": put_losses, "itemStyle": {"color": "#2ca02c"}},
    ]
  }

  # chart2：未平倉合約數 (維持不變)
  call_oi = [int(df_calls.set_index('strike').openInterest.get(s, 0)) for s in strikes]
  put_oi = [-int(df_puts.set_index('strike').openInterest.get(s, 0)) for s in strikes]

  chart2 = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["Call OI", "Put OI"]},
    "xAxis": {
      "type": "category",
      "data": strike_labels,
      "name": "履約價",
      "axisLabel": {"rotate": 45}
    },
    "yAxis": {
      "type": "value",
      "name": "Open Interest",
      "min": "dataMin",
      "max": "dataMax"
    },
    "series": [
      {"name": "Call OI", "type": "bar", "stack": "x", "data": call_oi, "itemStyle": {"color": "#d62728"}, "markLine": mark_line},
      {"name": "Put OI", "type": "bar", "stack": "x", "data": put_oi, "itemStyle": {"color": "#2ca02c"}},
    ]
  }

  del call_losses, put_losses, call_oi, put_oi
  gc.collect()

  return json.dumps({"chart1": chart1, "chart2": chart2})




################################################################################################################################################################
@app.route('/maxpain/', methods=['GET', 'POST'])
def maxpain():
  ticker = ""
  expirations = []
  selected_exp = ""
  max_pain = None
  chart = None
  error = None
  underlying_price = None

  if request.method == 'POST':
    action = request.form.get('action')
    ticker = request.form.get('ticker', '').upper()
    selected_exp = request.form.get('expiration')

    try:
      if action == 'get_expirations':
        expirations = get_expirations(ticker)

      elif action == 'get_chart':
        expirations = get_expirations(ticker)               
        if not selected_exp:
            raise ValueError("請選擇到期日")
        calls, puts = get_option_chain(ticker, selected_exp)
        max_pain = calculate_max_pain(calls, puts)
        
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            underlying_price = 0
        else:
            underlying_price = hist['Close'][-1]
          
        del hist
        gc.collect()
      
        chart = build_chart_option(calls, puts, ticker, max_pain, underlying_price)

    except Exception as e:
      error = str(e)

  print(f"max_pain_price = {max_pain}, underlying_price = {underlying_price}")

  return render_template(
    'maxpain.html',
    ticker=ticker,
    expirations=expirations,
    selected_exp=selected_exp,
    max_pain=max_pain,
    underlying_price=underlying_price,
    chart=chart,
    error=error
  )




################################################################################################################################################################
################################################################################################################################################################
@app.route('/tgs/')
def index():
  return render_template('tgs.html')




################################################################################################################################################################
@app.route('/tgs_analyze', methods=['POST'])
def analyze():
  data = request.form
  
  # 1. 獲取參數
  user_api_key = data.get('api_key')
  if user_api_key == "wwsspp":
    user_api_key = api_key
  selected_model = data.get('model_name')

  if not user_api_key:
    return jsonify({'success': False, 'error': '請輸入有效的 Gemini API Key'})
  
  # 3. 獲取策略參數
  company_name = data.get('company_name')
  context_structure = data.get('context_structure')
  context_drivers = data.get('context_drivers')
  context_uncertainty = data.get('context_uncertainty')
  boundary_conditions = data.get('boundary_conditions')

  # 4. Prompt (邏輯不變，維持 3C 架構)
  prompt = f"""
  你現在是精通李吉仁教授《轉型再成長》一書的首席策略顧問。
  你的任務是為 **「{company_name}」** 這家公司進行深度的策略規劃。
  請使用你的 Google Search 搜尋能力，先對該公司做深度研究，再依據 **Context (環境脈絡)、Change (策略改變)、Choice (策略選擇)** 的 3C 架構進行分析。
  
  *** 深度研究指令 (Deep Research Instructions) ***
  1.  **廣泛搜尋**：請不要只進行一次搜尋。請利用 Google Search 工具，針對該公司的「財務報表」、「競爭對手動態」、「產業分析報告」與「最新新聞」進行多角度的資料檢索。
  2.  **數據支撐**：分析時，請務必引用具體的數字（如營收成長率、毛利率變化、市佔率）來支持你的論點。
  3.  **交叉比對**：請結合搜尋到的外部客觀數據，與使用者提供的內部 Context 進行交叉比對。
  
  **使用者輸入 (Context)：**
  1. 目標公司：{company_name}
  2. 產業結構與改變脈絡: {context_structure}
  3. 未來成長驅動因子: {context_drivers}
  4. 不確定因素與可變性: {context_uncertainty}
  5. 邊界條件: {boundary_conditions}

  ---
  **任務執行步驟：**

  ### 第一部分：財務與成長動力掃描 (基於搜尋結果)
  請搜尋 **{company_name}** 過去五年的財務報表與新聞，簡要分析：
  * **營收與獲利趨勢：** (近五年是成長、持平還是衰退？)
  * **主要成長/衰退原因：** (市場因素或競爭因素？)
  * **現有核心動力：** (目前是靠什麼賺錢？)
  
  ### 第二部分：3C 策略架構分析
  基於上述財務背景與使用者的輸入，進行 Context, Change, Choice 分析：
  
  **Module 1: Context 情境洞察**
  * 根據使用者輸入，辨識成長機會與形成成長機會的結構性脈絡，同時理解未來可能的風險，作為後續成長方向與路徑選擇的依據。
  
  **Module 2: Change 變革核心**
  * 基於**改變以創造未來**的核心概念，建立想要改變的方向：建立事業新願景，提升價值定位，建構新競爭優勢。

  **Module 3: Choice 策略選擇**
  * 首先根據使用者輸入內容，分析**企業核心能力**
  * 根據**由內而外**與**由外而內**的兩種策略思維，結合**企業核心能力**，回答以下的關鍵問題 ：產品市場選擇，商業模式選擇，成長模式選擇 (外部併購、內部發展、策略性外包、切割獨立)。

  ### 第三部分：轉型再成長的策略擬定
  * 最後，根據第一部分的財務與成長動力掃描以及第二部分的3C策略架構分析，用**以終為始**的心智模式，分析企業領導人應該建立的企業願景。 
  * 接下來，透過願景建立新的期望目標，盤點現狀與期望目標間的差距，發展關鍵路徑，幫此公司擬訂**轉型再成長**的策略。
  * 策略的規劃需要按照書中的 **SPTSi** 架構：
    1. **Strategy Choice** 從 Gap Analysis 建立若干策略軸線。
    2. 在特定策略軸線下，**Key Path** 符合 MECE 原則拆解，確保所有路徑匯聚起來可以造成策略軸線想要改變的結果。
    3. **Tactical Action** 需對應所列出的 **Key Path**，專注於兩類戰術行動：一種是改變現狀的行動，另外一種是攸關重要資源投入的專案行動。
    4. **Success Indicator** 需對應所列出的 **Tactical Action**，可以有兩種不同面向的界定：一種是兼容過程與結果指標，另外一種是兼容品質與數量的成功指標。
  """

  """
  # 2. 設定 Gemini
  try:
    genai.configure(api_key=user_api_key)
    # 使用使用者選擇的模型
    model = genai.GenerativeModel(selected_model)
  except Exception as e:
    return jsonify({'success': False, 'error': f'模型設定失敗: {str(e)}'})

  try:
    response = model.generate_content(prompt)
    analysis_html = markdown.markdown(response.text, extensions=['fenced_code'])
    #return jsonify({'success': True, 'content': analysis_html, 'raw_markdown': response.text})
    return jsonify({'success': True, 'content': analysis_html})
  except Exception as e:
    return jsonify({'success': False, 'error': str(e)})
  """
  
  url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
  
  headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": user_api_key
  }

  # ★★★ 關鍵邏輯：根據模型版本切換工具名稱 ★★★
  # Gemini 2.0 使用 "google_search"
  tool_definition = {"google_search": {}}
  payload = {
    "contents": [{
      "parts": [{"text": prompt}]
    }],
    "tools": [
      tool_definition
    ]
  }

  try:
    # 5. 發送請求
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    
    # 6. 錯誤處理
    if response.status_code != 200:
      error_msg = f"API Error ({response.status_code}): {response.text}"
      print(error_msg)
      return jsonify({'success': False, 'error': error_msg})

    # 7. 解析 JSON 回傳
    result_json = response.json()
    
    # 檢查是否有候選回應
    if 'candidates' not in result_json or not result_json['candidates']:
      return jsonify({'success': False, 'error': 'AI 未回傳任何內容 (可能被安全機制阻擋)'})
      
    candidate = result_json['candidates'][0]
    
    # 提取文字內容
    if 'content' in candidate and 'parts' in candidate['content']:
      parts = candidate['content']['parts']
        
      # 使用 List Comprehension 提取所有 part 的 text 並串接
      # part.get('text', '') 確保萬一某個 part 沒有 text 欄位也不會報錯
      raw_text = "".join([part.get('text', '') for part in parts])
      
      # 轉換 Markdown
      #analysis_html = markdown.markdown(raw_text, extensions=['fenced_code'])
      #print(analysis_html)
      
      return jsonify({
        'success': True, 
        'raw_markdown': raw_text
      })
    else:
       return jsonify({'success': False, 'error': '回傳格式異常，找不到 content parts'})

  except Exception as e:
    print(f"Server Error: {e}")
    return jsonify({'success': False, 'error': f"伺服器內部錯誤: {str(e)}"})




################################################################################################################################################################
################################################################################################################################################################
# --- 設定與常數 ---
TWSE_URL = "https://heatmap.fugle.tw/api/heatmaps/IX0001"
OTC_URL = "https://heatmap.fugle.tw/api/heatmaps/IX0043"

INDUSTRY_MAP = {
  "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
  "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業",
  "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
  "15": "航運業", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨",
  "19": "綜合", "20": "其他", "21": "化學工業", "22": "生技醫療業",
  "23": "油電燃氣業", "24": "半導體業", "25": "電腦及週邊設備業",
  "26": "光電業", "27": "通信網路業", "28": "電子零組件業",
  "29": "電子通路業", "30": "資訊服務業", "31": "其他電子業",
  "32": "文化創意業", "33": "農業科技業", "34": "電子商務",
  "35": "綠能環保", "36": "數位雲端", "37": "運動休閒",
  "38": "居家生活", "80": "管理股票",
}

# --- 全域資料快取 (Server-side Caching) ---
# 用於儲存資料，避免每次前端請求都打外部API，同時確保資料不超過5分鐘
DATA_CACHE = {
  "twse": None,
  "otc": None,
  "last_update": 0
}
CACHE_DURATION = 300  # 5分鐘 (秒)

# --- 輔助函式 ---

def industry_label(code) -> str:
  if code is None:
    return "未知產業"
  s = str(code).strip()
  if s.isdigit():
    s = s.zfill(2)
  return INDUSTRY_MAP.get(s, "未知產業")

def fetch_data_with_cache():
  """檢查快取，若過期則重新抓取資料"""
  now = time.time()
  # 如果資料是空的 或者 距離上次更新超過 5 分鐘
  if (DATA_CACHE["twse"] is None) or (now - DATA_CACHE["last_update"] > CACHE_DURATION):
    try:
      print(f"[{time.ctime()}] Fetching new data from Fugle API...")
      r_twse = requests.get(TWSE_URL, timeout=10)
      r_otc = requests.get(OTC_URL, timeout=10)
      
      if r_twse.status_code == 200:
        DATA_CACHE["twse"] = pd.DataFrame(r_twse.json().get("data", []))
      if r_otc.status_code == 200:
        DATA_CACHE["otc"] = pd.DataFrame(r_otc.json().get("data", []))
        
      DATA_CACHE["last_update"] = now
    except Exception as e:
      print(f"Error fetching data: {e}")
      # 如果抓取失敗，暫時保持舊資料或回傳 None
      pass

def build_treemap_figure(df: pd.DataFrame, type_filter: str, area_choice: str):
  """建立 Plotly 圖表物件 (邏輯同原 Dash 程式)"""
  if df is None or df.empty:
    return px.treemap(title="無資料或讀取失敗")

  data = df[df["type"] == type_filter].copy()
  
  if data.empty:
    # 空資料處理
    return px.treemap(title=f"{type_filter} Treemap（資料為空）")

  # 處理標籤與數值
  if type_filter == "INDEX":
    data["group_label"] = "各類指數"
    data["label"] = data["name"]
    value_key = "tradeValue"
  else:
    data["group_label"] = data["industry"].apply(industry_label)
    data["label"] = data.apply(lambda r: f'{r["name"]}({r["symbol"]})', axis=1)
    value_key = area_choice if area_choice in ("tradeValueWeight", "marketValueWeight") else "tradeValueWeight"

  data["value"] = pd.to_numeric(data.get(value_key), errors="coerce")
  data = data[data["value"].notna() & (data["value"] > 0)]
  
  # 再次檢查過濾後是否為空
  if data.empty:
     # 這裡簡化處理，若過濾後為空直接回傳空圖
     return px.treemap(title=f"{type_filter} 無有效數據")

  data["chg"] = pd.to_numeric(data.get("changePercent"), errors="coerce")

  fig = px.treemap(
    data,
    path=["group_label", "label"],
    values="value",
    color="chg",
    color_continuous_scale="RdYlGn_r",
    color_continuous_midpoint=0
  )

  fig.update_traces(tiling=dict(packing="binary"))

  # 處理 Hover 與 Text
  for col in ("closePrice", "changePercent"):
    if col not in data.columns:
      data[col] = None
      
  # 注意：這裡將 DataFrame 轉為 numpy array 傳入 customdata，
  # Plotly JSON 序列化時需要確保數據格式乾淨
  '''
  fig.update_traces(
    customdata=data[["closePrice", "changePercent"]].to_numpy(),
    textinfo="none",
    texttemplate="%{label}<br>%{customdata[0]:,.2f} | %{customdata[1]:+.2f}%",
    hovertemplate=(
      "<b>%{label}</b><br>"
      "產業: %{currentPath}<br>"
      "收盤: %{customdata[0]:,.2f}<br>"
      "漲跌幅: %{customdata[1]:+.2f}%<br>"
      f"面積({value_key}): %{{value:,.4f}}"
      "<extra></extra>"
    )
  )
  '''
  fig.update_traces(
    customdata=data[["closePrice", "changePercent", "group_label"]].to_numpy(),
    textinfo="none",
    texttemplate=(
      "<span style='font-size: 16px; font-weight:bold'>%{label}</span>"
      "<br>"
      "<span style='font-size: 12px'>%{customdata[0]:,.2f} | %{customdata[1]:+.2f}%</span>"
    ),
    hovertemplate=(
      # 1. 股票名稱：用 span 包起來，設大一點 (24px)
      "<span style='font-size: 16px; font-weight:bold'>%{label}</span><br>"
      
      # 2. 其他資訊：統一包在另一個 span 裡，或是每一行單獨設定 (這裡示範設為 16px)
      "<span style='font-size: 16px'>"
      "產業: %{customdata[2]}<br>"
      "收盤: %{customdata[0]:,.2f}<br>"
      "漲跌幅: %{customdata[1]:+.2f}%<br>"
      f"面積({value_key}): %{{value:,.4f}}"
      "</span>"
      
      # 3. 隱藏旁邊預設的標籤
      "<extra></extra>"
    )
  )

  fig.update_layout(
    margin=dict(t=40, l=0, r=0, b=0),
    coloraxis_colorbar=dict(title="漲跌幅(%)"),
    height=900, # 高度交給前端 CSS 控制，但這裡給個預設
    title=f"{'指數' if type_filter=='INDEX' else '股票'} Treemap（面積: {value_key}）"
  )
  return fig

# --- HTML 模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Taiwan Stock Heatmap</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: sans-serif; padding: 20px; background-color: #f9f9f9; }
    .controls { background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .tabs { display: flex; margin-bottom: 10px; border-bottom: 1px solid #ccc; }
    .tab-btn { 
      padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; 
      border-bottom: 3px solid transparent; transition: all 0.3s;
    }
    .tab-btn.active { border-bottom: 3px solid #007bff; color: #007bff; font-weight: bold; }
    .tab-btn:hover { background-color: #f0f0f0; }
    #chart-container { width: 100%; height: 100%; background: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .loading { color: #666; font-style: italic; margin-left: 10px; display: none;}
  </style>
</head>
<body>

  <h3>Taiwan Stock Heatmap</h3>
  
  <div class="controls">
    <div style="margin-bottom: 15px;">
      <label><strong>面積指標：</strong></label>
      <label><input type="radio" name="area_metric" value="tradeValueWeight" checked> 成交值權重</label>
      <label><input type="radio" name="area_metric" value="marketValueWeight"> 市值權重</label>
      <span id="loading-msg" class="loading">更新數據中...</span>
      <span style="float: right; font-size: 0.8em; color: #888;">每 5 分鐘自動更新</span>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('twse', 'INDEX', this)">上市指數</button>
      <button class="tab-btn" onclick="switchTab('twse', 'EQUITY', this)">上市個股</button>
      <button class="tab-btn" onclick="switchTab('otc', 'INDEX', this)">上櫃指數</button>
      <button class="tab-btn" onclick="switchTab('otc', 'EQUITY', this)">上櫃個股</button>
    </div>
  </div>

  <div id="chart-container"></div>

  <script>
    // 當前狀態
    let currentMarket = 'twse';
    let currentType = 'INDEX';

    // 初始化
    document.addEventListener('DOMContentLoaded', () => {
      updateChart();
      
      // 監聽 Radio Button 改變
      document.querySelectorAll('input[name="area_metric"]').forEach(radio => {
        radio.addEventListener('change', updateChart);
      });

      // 設定自動更新定時器 (5分鐘 = 300,000 毫秒)
      setInterval(updateChart, 300000);
    });

    function switchTab(market, type, btnElement) {
      // 更新狀態
      currentMarket = market;
      currentType = type;
      
      // 更新 UI 樣式
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btnElement.classList.add('active');
      
      // 重新抓取圖表
      updateChart();
    }

    async function updateChart() {
      const areaMetric = document.querySelector('input[name="area_metric"]:checked').value;
      const loadingMsg = document.getElementById('loading-msg');
      
      loadingMsg.style.display = 'inline';

      try {
        // 呼叫 Flask API
        const response = await fetch(`/twheatmap/api/get_chart?market=${currentMarket}&type=${currentType}&area=${areaMetric}`);
        const graphJson = await response.json();
        
        // 使用 Plotly 繪圖
        Plotly.newPlot('chart-container', graphJson.data, graphJson.layout, {responsive: true});
      } catch (error) {
        console.error("Error loading chart:", error);
        document.getElementById('chart-container').innerHTML = "<p style='padding:20px'>載入失敗，請稍後再試。</p>";
      } finally {
        loadingMsg.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""




################################################################################################################################################################
# --- Flask 路由 ---
@app.route("/twheatmap/")
def twheatmap_index():
  return render_template_string(HTML_TEMPLATE)




################################################################################################################################################################
@app.route("twheatmap/api/get_chart")
def twheatmap_get_chart():
  """API 端點：根據參數回傳 Plotly JSON"""
  # 取得參數
  market = request.args.get("market", "twse")  # twse 或 otc
  type_filter = request.args.get("type", "INDEX") # INDEX 或 EQUITY
  area_metric = request.args.get("area", "tradeValueWeight")

  # 確保資料是最新的 (快取檢查)
  fetch_data_with_cache()

  # 選擇對應的 DataFrame
  df = DATA_CACHE["twse"] if market == "twse" else DATA_CACHE["otc"]
  
  # 建立圖表
  fig = build_treemap_figure(df, type_filter, area_metric)
  
  # 將圖表轉換為 JSON 格式回傳給前端
  # 使用 plotly.utils.PlotlyJSONEncoder 確保格式正確
  return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)




################################################################################################################################################################
################################################################################################################################################################
if __name__ == '__main__':
    app.run(debug=True)
