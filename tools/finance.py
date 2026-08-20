import asyncio
from langchain.tools import tool
import yfinance as yf

@tool
async def stock_price(symbol: str):
    """Get stock price. Use NSE format like 'HDFCBANK.NS' for Indian stocks, or ticker symbol like 'AAPL' for US stocks."""
    def _fetch():
        x = yf.Ticker(symbol).history(period='1d')
        return {f'stock_price for {symbol}': x['Close']}

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        return {"error": f"Could not fetch stock price: {str(e)}"}