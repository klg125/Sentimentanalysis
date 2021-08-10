import difflib
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import matplotlib.pyplot as plt


#importing Opinion Lexicon Word Dictionary

f = open('bank2.txt')
n = open('neg.txt')
p = open ('pos.txt')

#converting the text files into lists readable in python

wordbank=[]

for i in f.read().split():
    wordbank.append(i)

negative=[]

for x in n.read().split():
    negative.append(x)


positive=[]

for y in p.read().split():
    positive.append(y)

#import ticker list
d=pd.read_csv('sp500.csv')
stocklist=d['tick']

tickerhigh=[]
for x in stocklist: 
    tickerhigh.append(x)

s=[]
#converting ticker list to lower-case - the format for difflib
[s.append(a.lower()) if not a.islower() else s.append(a) for a in tickerhigh]

s = list(set(s))

full=[]

#utilizing difflib to obtain 12 cloest matches 
for i in s:
    full.append(difflib.get_close_matches(i, wordbank, 12))

score=0
final=[]

#calculating total sentiment score
for i in full:
    for j in i:
        if j in negative:
            score-=1
        else:
            score+=1
    final.append(score/12)  #averaging total sentiment score
    score=0


#final2 = ((np.array(final) - np.mean(final)) / (np.max(final) - np.min(final))**3)*100
#put weight on super negative and super positive ticker sentiment scores
        


finallist=[]
supper=[]
#converting tickers back to upper case
[supper.append(a.upper()) if not a.isupper() else supper.append(a) for a in s]
#putting ticker and score side by side 
for i in range(0,len(final)): 
    finallist.append(supper[i])
    finallist.append(final[i]) 



df = pd.read_csv('SP500return.csv') #import SP500 stock returns
df.ticker=df.ticker.astype(str)




df1 = pd.DataFrame({'ticker': supper,
                    'sentiment': final})


datatop = df1[df1['sentiment']> df1['sentiment'].quantile(.8)]

databot=df1[df1['sentiment']< df1['sentiment'].quantile(.2)]


#bottom 20 percentile data
df2bot = pd.merge(df, databot,on=['ticker']) 


#converting given date to python's required data-time format
df2bot['newdate'] = df2bot['date'].map(lambda x: str(x)[:4]+'/'+str(x)[4:6]+'/'+str(x)[6:8]) 
df2bot['newdate'] = pd.to_datetime(df2bot['newdate']) 

df2bot['scaledrets'] = df2bot['return']*df2bot['sentiment'] #return times sentiment

#grouping by month and year
df2bot['months']= df2bot['newdate'].dt.month
df2bot['years']=df2bot['newdate'].dt.year
df3bot=df2bot.groupby(['years','months'],as_index=False).mean()



#top 20 percentile data


df2top = pd.merge(df,datatop,on=['ticker'])

df2top['newdate'] = df2top['date'].map(lambda x: str(x)[:4]+'/'+str(x)[4:6]+'/'+str(x)[6:8])
df2top['newdate'] = pd.to_datetime(df2top['newdate'])
df2top['scaledrets'] = df2top['return']*df2top['sentiment']
df2top['months']= df2top['newdate'].dt.month
df2top['years']=df2top['newdate'].dt.year
df3top=df2top.groupby(['years','months'],as_index=False).mean()




df4=df3top['scaledrets']-df3bot['scaledrets']



df = pd.read_csv('FFlist.csv')

FF=df[882:]

FF['newdate'] = FF['Date'].map(lambda x: str(x)[:4]+'/'+str(x)[4:6]+'/'+'01')
FF['newdate'] = pd.to_datetime(FF['newdate'])

FF['months']= FF['newdate'].dt.month
FF['years']=FF['newdate'].dt.year

dfreg=pd.merge(FF,df3bot,on=['months','years'])


X = dfreg[["Mkt-RF","SMB","HML"]] ## X usually means our input variables (or independent variables)
y = df4    ## Y usually means our output/dependent variable
X = sm.add_constant(X) #adding a constant


model = sm.OLS(y, X).fit() 
predictions = model.predict(X)


# Print out the statistics
#print(model.summary())


#for plotting multiple linear regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(FF['SMB'],FF['HML'],df4,c='blue', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
ax.set_xlabel('SMB')
ax.set_ylabel('HML')
ax.set_zlabel('scaledrets')
plt.show()
