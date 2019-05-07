import pandas_datareader as pdr

def stock_price(Company:str, start, end):
    '''Download Stock data from finance.yahoo'''
    df = pdr.DataReader(Company, 'yahoo', start, end)
    df.columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    return df 

