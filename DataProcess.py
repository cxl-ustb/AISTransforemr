import pandas as pd


df=pd.read_excel('./data/413719000/413719000.xls')

# lon:110.12347-110.2815  0.00063212
# lat: 20.023834-20.2712   0.000989464
# sog:0.0-14.9 0.0596
# cog：0.0-359 1.436

data=pd.concat([np.floor((df['经度']-df['经度'].values.min())/0.00063212),
                  np.floor((df['纬度']-df['纬度'].values.min())/0.000989464),
                  np.floor((df['速度']-df['速度'].values.min())/0.0596),
                  np.floor((df['对地航向']-df['对地航向'].values.min())/1.436)],axis=1)
data.to_pickle('./data/413719000/413719000.pkl')
