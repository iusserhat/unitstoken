from fastapi import FastAPI
import requests
import numpy as np
import pandas as pd
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

API_URL = "https://api.coingecko.com/api/v3/simple/price"
PARAMS = {"ids": "unit0", "vs_currencies": "usd"}

data = []  # Fiyatları saklayan liste
MAX_DATA_SIZE = 100  # Daha fazla geçmiş veri tut

async def get_price_data():
    try:
        response = requests.get(API_URL, params=PARAMS, timeout=2)
        response.raise_for_status()
        return response.json().get("unit0", {}).get("usd")
    except requests.RequestException:
        return None

def rsi_signal(prices, window=14):
    if len(prices) < window:
        return "HOLD"  # Yeterli veri yoksa 'HOLD' sinyali ver
    
    df = pd.DataFrame(prices, columns=['price'])
    
    # Fiyat değişimleri
    df['delta'] = df['price'].diff()
    df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
    df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)

    # Hareketli ortalamalar
    avg_gain = df['gain'].rolling(window=window).mean()
    avg_loss = df['loss'].rolling(window=window).mean()

    # RSI hesaplama
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    rsi_value = df['rsi'].iloc[-1]
    
    # RSI sinyalleri
    if rsi_value < 30:
        return "BUY"
    elif rsi_value > 70:
        return "SELL"
    else:
        return "HOLD"

@app.get("/predict")
async def predict():
    global data
    price = await get_price_data()
    if price:
        data.append(price)
        if len(data) > MAX_DATA_SIZE:
            data.pop(0)  # En eski veriyi at
        print(f"Veriler: {data[-10:]}")  # Son 10 veriyi yazdırarak kontrol et
        signal = rsi_signal(data)
        return {"price": price, "signal": signal}
    return {"error": "Veri alınamadı"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Units Token Al-Sat Sinyali</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body {
                background-color: #01010d;
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .card {
                background-color: #10dcb4;
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
            }
            .signal-buy { color: #4CAF50; font-weight: bold; }
            .signal-sell { color: #FF5252; font-weight: bold; }
            .signal-hold { color: #CCCCCC; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card text-center p-4">
                        <h2 class="mb-3">📈 Units Token Fiyat ve Sinyal</h2>
                        <h4>Güncel Fiyat: <span id="price">-</span> USD</h4>
                        <h5>Al-Sat Sinyali: <span id="signal">-</span></h5>
                        <button class="btn btn-primary mt-3" onclick="fetchData()" id="fetch-btn">Sinyal Al</button>
                        <p id="loading" style="display:none; color:yellow;">🔄 Sinyal hesaplanıyor...</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function fetchData() {
                $("#loading").show();
                $("#fetch-btn").prop("disabled", true);

                $.getJSON("/predict", function(data) {
                    $("#loading").hide();
                    $("#fetch-btn").prop("disabled", false);

                    if (data.error) {
                        $("#price").text("Veri alınamadı");
                        $("#signal").text("Bilinmiyor").removeClass().addClass("signal-hold");
                    } else {
                        $("#price").text(data.price.toFixed(2));
                        let signalClass = "signal-hold";
                        if (data.signal === "BUY") signalClass = "signal-buy";
                        else if (data.signal === "SELL") signalClass = "signal-sell";
                        $("#signal").text(data.signal).removeClass().addClass(signalClass);
                    }
                });
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
