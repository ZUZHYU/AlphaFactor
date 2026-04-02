import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('https://raw.githubusercontent.com/benckj/mpl_style/main/uzh.mplstyle')
from glob import glob
from datetime import datetime
from datetime import timedelta
import pickle


class alpha_Factor_Strategy:

    """
    Description of the price data:
    Under the data path (data_path), each token has a specific .csv file containning minute-level price data. 
    Columns include 'Close', 'Open Time','Volume'
    starting from its listing data on Binance to Febuary 15th, 2026.
    All token pairs' quote token is USDT.
    There are more than 400 token_USDT pairs upon loading. 
    In our case, data_path = '/local/scratch/yuzhang_utxo/token_price/'
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = glob(self.data_path+'new_*_new_minute_price.csv')

        self.token_listing_date = None
        with open(self.data_path+'/all_tokens_listing_time.csv','rb') as f:
            self.token_listing_date = pickle.load(f)

        self.select_tokens = None
        self.all_tokens_factors = None
        # if self.all_tokens_factors != None:
        #     self.factor_name_list = self.all_tokens_factors.columns
        self.factor_IC = None

    
    """
    This function is to select tokens based on tokens' listing date
    a token is selected if this token's listing data on Binance is earler than the parameter "early_listing_date".
    The default value of "early_listing_date" is '2020-01-01'
    """
    def select_token(self,early_listing_date = '2020-01-01'):
        
        self.selected_tokens = [k for k,v in self.token_listing_date.items() if v<datetime.strptime(early_listing_date,'%Y-%m-%d')]

        return self.selected_tokens


    """
    This function calculate 3 volume distribution factors at the same price with minute-level data of a token in a day.
    It return factor values and factors' names

    The data is a minute-level DataFrame in a day, including close price and volume columns.

    This function will be passed to cal_specific_token_factors as parameter "factor_cal_with_daily_data".
    """

    """
    We can write other factor functions here.
    """

    def same_price_volume_factor_cal(self, daily_data):

        close = daily_data['Round Close'].iloc[-1]
        high = np.nanmax(daily_data['Round Close'])
        low = np.nanmin(daily_data['Round Close'])

        daily_data = daily_data.groupby('Round Close')[['Volume']].sum()

        daily_data.reset_index(inplace=True)

        vol = daily_data['Volume'].sum()
        vol_thd = vol/2

        idx = np.argmax(daily_data['Volume'])
        daily_data['diff'] = np.abs(daily_data['Round Close']-daily_data.iloc[idx]['Round Close'])

        daily_data.sort_values(by = ['diff','Volume'],ascending=(1,0),inplace=True)

        #find the upper-bound and lower-bound
        current_vol = 0
        for i in range(len(daily_data)):
            current_vol += daily_data['Volume'].iloc[i]
            if current_vol > vol_thd:
                daily_data = daily_data.iloc[:i+1]
                break
        
        # upper_bound vsa_high
        vsa_high = daily_data['Round Close'].max()
        vsa_high2min = vsa_high/low

        # lower_bound vsa_low
        vsa_low = daily_data['Round Close'].min()
        vsa_low2max = vsa_low/high

        vsa_ratio = vsa_low/close

        # return factors
        return [vsa_ratio, vsa_high2min, vsa_low2max], ["vsa_ratio", "vsa_high2min", "vsa_low2max"]



    # Calculate factors for a specific data during a specific time period 

    def cal_specific_token_factors(self, token_data, token_name, factor_cal_with_daily_data, 
                                   start_date_str = '2024-12-01', end_date_str= '2026-02-01'):

        start_date = datetime.strptime(start_date_str,'%Y-%m-%d')
        end_date = datetime.strptime(end_date_str,'%Y-%m-%d')

        start_date_timestamp = int(start_date.timestamp()*1000)+60*1000   #starting from 00:01:00
        end_date_timestamp = int(end_date.timestamp()*1000)+60*1000       #ending at 24:00:00

        start_index = np.searchsorted(np.array(token_data['Open Time']),start_date_timestamp,side='left')
        end_index = np.searchsorted(np.array(token_data['Open Time']),end_date_timestamp,side='left')

        token_data_for_return = token_data.iloc[start_index:end_index+2*24*60]
        token_data_for_return.reset_index(inplace=True, drop=True)

        token_data = token_data.iloc[start_index:end_index]
        token_data.reset_index(inplace=True, drop=True)
        # The round number can be 2 or 3. A better way is to use binning method.
        token_data['Round Close'] = token_data['Close'].round(2)  
        # token_data.loc[:, 'Round Close'] = token_data['Close'].round(2)

        factor_val = []

        factor_name_list = None

        for i in range(int((end_date_timestamp-start_date_timestamp)/60/60/24/1000)):
            daily_data = token_data.iloc[i*24*60:(i+1)*24*60]

            return_rate = token_data_for_return.iloc[(i+2)*24*60]['Close']/token_data_for_return.iloc[(i+1)*24*60]['Close'] -1

            daily_factor_values, daily_factor_names = factor_cal_with_daily_data(daily_data)

            if factor_name_list == None:
                factor_name_list = daily_factor_names

            factor_val.append([start_date+timedelta(days=i)]+daily_factor_values+[return_rate])

        token_factor = pd.DataFrame(data = factor_val, columns=['date']+factor_name_list+['return_rate'])
        token_factor['token'] = token_name

        # token_factor = token_factor.set_index(['date', 'token']).sort_index()
        return token_factor



    """
    'factor_function' is the function that is used to calculate the factor of a token using its minute-level price 
    data on a specific day. It return the factor value

    We can use 'same_price_volume_factor_cal' in our example, and we can also use other factor calculation functions.
    """
    def factor_cal(self,factor_function, start_date_str = '2024-12-01', end_date_str= '2026-02-01'):

        self.all_tokens_factors = pd.DataFrame()
        
        for token in self.selected_tokens:

            token_data = pd.read_csv(self.data_path+'new_'+token+'_new_minute_price.csv', header = 0)
            token_factor = self.cal_specific_token_factors(token_data,token,factor_function,
                                                           start_date_str = '2024-12-01', end_date_str= '2026-02-01')

            self.all_tokens_factors = pd.concat([self.all_tokens_factors,token_factor])

        return self.all_tokens_factors
        
        
        


    """ 
    Calculate IC mean, ICIR and t-test statistics
    """
    def calc_ic(self,factor_name_list):
        factor_dict = {}
        df = self.all_tokens_factors.copy()
        for factor in factor_name_list:
            ic = df.groupby('date').apply(
                lambda x: x[factor].corr(x['return_rate'], method='spearman'))
            
            factor_dict[factor+'_ic'] = ic

        self.factor_IC = pd.DataFrame(factor_dict)

        for factor in factor_name_list:
            ic_mean = self.factor_IC[factor+'_ic'].mean()
            ic_std = self.factor_IC[factor+'_ic'].std()
            icir = ic_mean/ic_std
            t_stat = icir*np.sqrt(len(self.factor_IC))

            print(factor, 'IC:', ic_mean, 'ICIR:', icir, 't-statistics:',t_stat,)

        return self.factor_IC

    # Plot the factor IC and factor's cummulative IC
    def plot_ic(self, factor_name):

        if self.factor_IC is not None and factor_name not in self.factor_IC.columns:
            self.calc_ic([factor_name])

        self.factor_IC[factor_name+'_cum_ic'] = self.factor_IC[factor_name+'_ic'].cumsum()
        
        ax1 = self.factor_IC[factor_name+'_cum_ic'].plot(figsize=(12,5), color = 'C1')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(factor_name+'_cum_ic')


        ax2 = ax1.twinx()
        self.factor_IC[factor_name+'_ic'].plot(figsize=(12,5),ax=ax2, color = 'C0')
        ax2.set_ylabel(factor_name+'_ic')
        plt.legend()
        plt.show()
                    


    """ 
    Test the monotonicity of the factor in different groups.
    df is the dataframe that include the factor value of each token on each date.
    """
    def test_monotonicity(self, factor, n_groups=5):
        df = self.all_tokens_factors.copy()
        # rank into quantiles per date
        df['rank'] = df.groupby('date')[factor].rank(method='first',pct=True)
        
        df['group'] = df.groupby('date')['rank'].transform(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop'))
        
        # average return per group per date
        return_by_group = df.groupby(['date', 'group'])['return_rate'].mean().unstack()
        
        # average across time
        mean_return_by_group = return_by_group.mean()
        
        return return_by_group, mean_return_by_group

    """
    Backtesting of factor. The input data (df) contain all tokens' factor values on each date.
    """
    def backtest_factor(self, factor, n_groups=5):
        df = self.all_tokens_factors.copy()
        
        # rank into quantiles
        df['rank'] = df.groupby('date')[factor].rank(method = 'first',pct=True)
        
        df['group'] = df.groupby('date')['rank'].transform(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
        )
        
        # long top group, short bottom group
        long = df[df['group'] == n_groups - 1].groupby('date')['return_rate'].mean()
        short = df[df['group'] == 0].groupby('date')['return_rate'].mean()
        
        ls_ret = (long - short)/2
        
        # cumulative return
        cum_ret = (1 + ls_ret).cumprod()
        
        # metrics
        sharpe = ls_ret.mean() / ls_ret.std() * np.sqrt(252)
        max_dd = -(1-cum_ret / cum_ret.cummax()).max()
        
        result = {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': cum_ret.iloc[-1] - 1
        }
        
        return ls_ret, cum_ret, result


    def plot_motonicity_backtest(self, factor_name_list):

        if factor_name_list != None:
            for factor in factor_name_list:
                group_ret, mean_group_ret = self.test_monotonicity(factor)

                fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12, 12))
                for col in group_ret.columns:
                    ax1.plot(group_ret.index, group_ret[col], label=f'Group {col}')

                ax1.legend()
                ax1.set_title('Daily Group Returns With Factor '+factor)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Return')
                ax1.axhline(0, linestyle='--')
                    
                for col in group_ret.columns:
                    ax2.plot(group_ret.index, group_ret[col].cumsum(), label=f'Group {col}')
                ax2.legend()
                ax2.set_title('Cummulative Group Returns Over Time With Factor '+factor)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Cummulative Return')
                ax2.axhline(0, linestyle='--')
                plt.show()


                ls_ret, cum_ret, stats = self.backtest_factor(factor)
                print(stats)
                plt.figure(figsize = (12,5))
                cum_ret.plot(figsize=(10,4))
                plt.title('Return of Long-Short Strategy Over Time '+factor)
                plt.axhline(1, linestyle='--')
                plt.ylabel('Strategy Return')
                plt.show()

    """
    Return all factors' names
    """
    def factor_names(self):
        if self.all_tokens_factors != None:
            return self.all_tokens_factors.columns
