from LoadData import stock_price
import pandas as pd

class get_FundamentalIndicators(object):
    
    def __init__(self, df):
        self.df = df

    # VIX
    def add_VIX(self):
        Company = '^VIX'
        VIX = stock_price(Company, "1992-12-31", "2018-12-31")
        VIX.rename(columns = {'Close':'VIX_Close'}, inplace = True)
        self.df = self.df.join(VIX.VIX_Close)
        return self
    
    # S&P close and volume
    def add_SnP(self):
        Company = '^GSPC'
        SnP = stock_price(Company, "1992-12-31", "2018-12-31")
        SnP.rename(columns = {'Close':'S&P Close', 'Volume':'S&P Volume'}, inplace = True)
        self.df = self.df.join(SnP.loc[:,['S&P Close', 'S&P Volume']])
        return self
        
    # DXY
    def add_DXY(self):
        DXY1981 = pd.read_csv('../Final/Data/DXY1981.csv')
        DXY = pd.read_csv('../Final/Data/DXY.csv')
        DXY = pd.concat([DXY1981, DXY]).drop_duplicates('Date').reset_index(drop = True)
        DXY.Date = pd.to_datetime(DXY.Date)
        DXY = DXY.set_index('Date')
        DXY.rename(columns = {'Price':'DXY'}, inplace=True)
        DXY.sort_index
        self.df = self.df.join(DXY.DXY)
        return self

    # Crude Oil
    def add_Oil(self):
        url = "https://www.quandl.com/api/v3/datasets/CHRIS/CME_CL1.csv"
        wticl1 = pd.read_csv(url, index_col = 0, parse_dates = True)
        wticl1.sort_index(inplace = True)
        names = []
        for i in wticl1.columns:
            names += [i+'_oil']
        wticl1.columns = names
        wticl1 = wticl1.loc[:,['Last_oil','Volume_oil']]
        self.df = self.df.join(wticl1)
        return self
        
    # US bond rates
    def add_USbond(self):
        US_bond1981 = pd.read_csv('../Final/Data/USA_10Y_bong_1981.csv')
        US_bond = pd.read_csv('../Final/Data/USA_10Y_bong.csv')
        US_bond = pd.concat([US_bond1981, US_bond]).drop_duplicates('Date').reset_index(drop = True)
        US_bond.Date = pd.to_datetime(US_bond.Date)
        US_bond = US_bond.set_index('Date')
        US_bond.rename(columns = {'Price':'US_bond'}, inplace = True)
        US_bond.sort_index
        self.df = self.df.join(US_bond.US_bond)
        return self
    
    def modify(self):
        return self.df
    
def get_FI_for_analysis(df):
    '''Fundamental indicators'''
    df = get_FundamentalIndicators(df).add_VIX()\
                                      .add_SnP()\
                                      .add_DXY()\
                                      .add_Oil()\
                                      .add_USbond()\
                                      .modify()
    df.dropna(inplace = True)
    df.drop(['High', 'Low', 'Open', 'Adj Close'], axis=1, inplace=True)
    return df        

def get_FI_for_model(df):
    '''Fundamental indicators'''
    df = get_FundamentalIndicators(df).add_DXY()\
                                      .modify()
#     .add_VIX()\
    return df