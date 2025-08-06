from flask import Flask, request, render_template, Response
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




app = Flask(__name__)

# Initialization
api_key = os.environ.get('API_KEY')
cm_url = os.environ.get('CM_URL')
cm_url2 = os.environ.get('CM_URL2')
si_url = os.environ.get('SI_URL')
tw_sf_url = os.environ.get('TW_SF_URL')


use_ollama = False
ollama_model = "deepseek-r1:8b"

BARS = 200




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
    headers_benzinga = {
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
      df = pd.concat(dfs, axis=1)
      df.index = pd.to_datetime(df.index, unit='ms').strftime('%Y-%m-%d')
      df.rename(columns={"'融資餘額(張)'": "BB", "'融券餘額(張)'": "SB", "'借券賣出餘額(張)'": "LSB"}, inplace=True)

      df_reset = df.tail(BARS).reset_index()
      return_value = df_reset.to_dict(orient='records')

  return return_value




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
  
  options = stock.options[:8]
  options_data = {}
  for date in options:
    opt = stock.option_chain(date)
    options_data[date] = {
      "calls": opt.calls.round(4).to_dict(orient="records"),
      "puts": opt.puts.round(4).to_dict(orient="records")
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
      ]
    }
    response = requests.post(url, headers=headers, json=data, timeout=600)
    if response.status_code == 200:
      result = response.json()
      # 取出回應內容
      return result['candidates'][0]['content']['parts'][0]['text']
    else:
      raise Exception(f"❌ Gemini API error: {response.status_code} {response.text}")




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
        prompt_prefix = f'請根據{ticker}的歷史股價與技術分析（含10MA， 20MA， 60MA， 200MA， RSI， ATR， Volume， MACD Histogram (MACDH)， 60MA Bollinger Band (BBU， BBD)， 200MA Diff Z-Score (200MADZ)）配合對應的成交量 (Volume)，財報 (financials， quarterly_financials， cash_flow， quarterly_cashflow， info)，與期權資料，{additional_prompt}，列出近期財報亮點與分析師評論 (upgrades_downgrades， eps_trend， revenue_estimate) 的整理，且產生一份繁體中文個股分析報告，首先列出目前價格與關鍵支持價位以及根據財報預測數據所推算的未來股價，然後內容包含基本面 (數字要有YoY加減速的分析，以及free cashflow的研究，並且根據年度財報預估與當季累積財報數字，預估後面一兩季的營收獲利起伏與對應的PE PS PB ratio，並且以表格列出每季EPS與營收增減的速度與加速度)、技術面 (配合成交量分析， 例如是否有價量背離) 與期權市場的觀察與建議。 若資料中有台灣股市 (mainforce_tw) 主力當日買賣超 (mf)，主力買賣超累積 (mf_acc)，買賣家差數 (b_s)，順便分析主力吃或出貨狀況。若資料中有short_stats，根據SF (short floating) 與SR (short ratio) 分析市場空單狀況及嘎空可能性。若資料中有台灣股市 (securities_financing_tw) 融資餘額 (BB)， 融券餘額 (SB)， 借券賣出餘額 (LSB)，分析市場空單狀況及嘎空可能性。'
        prompt = f'{prompt_prefix}\n資料如下：\n{stock_data}'
        #print('----------------------------------------')
        #print(prompt)
        #print('----------------------------------------')
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
if __name__ == '__main__':
    app.run(debug=True)
