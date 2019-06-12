import pandas as pd

class get_TechnicalIndicators(object):
    
    def __init__(self, df):
        self.df = df.reset_index()

    # Create MACD
    def add_MACD(self):
        ema_ts = self.df['Close'].ewm(span=26).mean()  
        ema_t = self.df['Close'].ewm(span=12).mean()  
        MACD = (ema_t-ema_ts).rename('MACD')
        self.df = self.df.join(MACD)
        return self
    
    #Stochastic Oscillator
    def add_StochasticOscillator(self, window=14):
        L14 = self.df['Low'].rolling(window=window).min()
        H14 = self.df['High'].rolling(window=window).max()
        SO = (100*((self.df['Close'] - L14) / (H14 - L14))).rename('%K')
        self.df = self.df.join(SO)
        return self
    
    # Ease of Movement 
    def add_EVM(self, ndays=14): 
        dm = ((self.df['High'] + self.df['Low'])/2) - ((self.df['High'].shift(1) + self.df['Low'].shift(1))/2)
        br = (self.df['Volume'] / (max(self.df['Volume'])//100)) / ((self.df['High'] - self.df['Low']))
        EVM = dm / br 
        EoM = EVM.rolling(window=ndays).mean().rename('EoM')
        self.df = self.df.join(EoM)
        return self

    # Force Index 
    def add_ForceIndex(self, ndays=1): 
        FI = (self.df['Close'].diff(ndays) * self.df['Volume']).rename('ForceIndex')
        self.df = self.df.join(FI)
        return self


    #Average True range
    def add_AverageTrueRange(self, ndays=14):
        i = 0
        TR_l = [0]
        while i < self.df.index[-1]:
            TR = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Close']) - min(self.df.loc[i + 1, 'Low'], self.df.loc[i,'Close'])
            TR_l.append(TR)
            i = i + 1
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=ndays, min_periods=ndays).mean(), name='ATR_' + str(ndays))
        self.df = self.df.join(ATR)
        return self

    #Mass index
    def add_MassIndex(self):
        Range = self.df['High'] - self.df['Low']
        EX1 = Range.ewm(span=9, min_periods=9).mean()
        EX2 = EX1.ewm(span=9, min_periods=9).mean()
        Mass = EX1 / EX2
        MassIndex = pd.Series(Mass.rolling(25).sum(), name='Mass Index')
        self.df = self.df.join(MassIndex)
        return self
    
    #STD 14
    def add_StandardDeviation(self, ndays=14):
        self.df = self.df.join(pd.Series(self.df['Close'].rolling(ndays, min_periods=ndays).std(), name='STD_' + str(ndays)))
        return self
    
    #MFI
    def add_MoneyFlowIndex(self, ndays = 14):
        PP = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        i = 0
        PosMF = [0]
        while i < self.df.index[-1]:
            if PP[i + 1] > PP[i]:
                PosMF.append(PP[i + 1] * self.df.loc[i + 1, 'Volume'])
            else:
                PosMF.append(0)
            i = i + 1
        PosMF = pd.Series(PosMF)
        TotMF = PP * self.df['Volume']
        MFR = pd.Series(PosMF / TotMF)
        MFI = pd.Series(MFR.rolling(ndays, min_periods = ndays).mean(), name='MFI_' + str(ndays))
        self.df = self.df.join(MFI)
        return self
    
    def modify(self):
        self.df.set_index('Date', inplace=True)
        self.df.dropna(inplace=True)
        self.df.drop(['High', 'Low', 'Open', 'Adj Close'], axis=1, inplace=True)
        return self.df
    

def get_TI(df):
    '''Technical indicators generation'''
    df = get_TechnicalIndicators(df).add_MACD()\
                                    .add_StochasticOscillator()\
                                    .add_EVM()\
                                    .add_ForceIndex()\
                                    .add_MassIndex()\
                                    .add_AverageTrueRange()\
                                    .add_MoneyFlowIndex()\
                                    .add_StandardDeviation\
                                    .modify()
    
    return df