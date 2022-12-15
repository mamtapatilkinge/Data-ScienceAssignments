#!/usr/bin/env python
# coding: utf-8

# ### Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &amp;
# ### comment about the values / draw inferences, for the given dataset
# ### - For Points,Score,Weigh&gt;
# ### Find Mean, Median, Mode, Variance, Standard Deviation, and Range
# ### and also Comment about the values/ Draw some inferences.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


# In[2]:


dir(pd)


# In[3]:


help(pd.read_csv)


# In[4]:


df=pd.read_csv('V:\Mamta\CSVFiles\Q7.csv')


# In[5]:


df.mean()


# In[6]:


df.median()


# In[7]:


df.mode()


# In[8]:


df.var()


# In[9]:


df.std()


# In[10]:


df.min()


# In[11]:


df.max()


# In[12]:


plt.boxplot(df.Points, vert=False)
plt.boxplot(df.Score, vert=False)
plt.boxplot(df.Weigh, vert=False)
sns.jointplot(df.Points, kind='kde')
sns.jointplot(df.Score, kind='kde')
sns.jointplot(df.Weigh, kind='kde')


# In the statistics, the range is the spread of your data from the lowest to the highest value in the distribustion.
# 
# 

# In[13]:


import matplotlib.pyplot as plt
import numpy as np

plt.title("Mean,Median,variance,std.deviation value's")
plt.xlabel("Value")
plt.ylabel("point score weigh")

point = np.array([3.5965,3.695,0.2858,0.5346])
score = np.array([3.2172,3.325,0.9573,0.9784])
weigh = np.array([17.8487,17.71,3.1931,1.7869])

plt.plot(point)
plt.plot(score)
plt.plot(weigh)

plt.show()


# ### Q9)  a) Calculate Skewness, Kurtosis & draw inferences on the following data Cars speed and distance

# In[14]:


dir(pd)


# In[15]:


help(pd.read_csv)


# In[16]:


df=pd.read_csv('V:\Mamta\CSVFiles\Q9_a (2).csv')


# In[17]:


df.skew()


# In[18]:


df.kurtosis()


# ### SP and Weight(WT)
# ### Use Q9_b.csv

# In[19]:


dir(pd)


# In[20]:


help(pd.read_csv)


# In[21]:


df=pd.read_csv('V:\Mamta\CSVFiles\Q9_b (1).csv')


# In[22]:


df.skew()


# In[23]:


df.kurtosis()


# ### Q11) Suppose we want to estimate the average weight of an adult male in    Mexico. We draw a random sample of 2,000 men from ### a population of 3,000,000 men and weigh them. We find that the average person in our sample weighs 200 pounds, and the ### standard deviation of the sample is 30 pounds. Calculate 94%,98%,96% confidence interval 

# In[24]:


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


# In[25]:


# Average weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))


# In[26]:


# Average weight of Adult in Mexico with 98% CI
stats.norm.interval(0.98,200,30/(2000**0.5))


# In[27]:


# Average weight of Adult in Mexico with 96% CI
stats.norm.interval(0.96,200,30/(2000**0.5))


# ### Q12) Below are the scores obtained by a student in tests  
# 
# ### 34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56 
# 
# ### 1) Find mean, median, variance, standard deviation. 
# 
# ### 2) What can we say about the student marks?  

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


x=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])


# In[30]:


# Mean
x.mean()


# In[31]:


# Median
x.median()


# In[32]:


# Variance
x.var()


# In[33]:


plt.boxplot(x)


# ### Q 20) Calculate probability from the given dataset for the below cases
# 
# ### Data _set: Cars.csv Calculate the probability of MPG of Cars for the below cases. MPG <- Cars$MPG 
# a. P(MPG>38) b. P(MPG<40) c. P (20<MPG<5

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


# In[35]:


dir(pd)


# In[36]:


help(pd.read_csv)


# In[37]:


df=pd.read_csv('V:/Mamta/CSVFiles/Cars.csv')


# In[38]:


df.mean()


# In[39]:


df.std()


# In[40]:


# P(MPG>38
#1-stats.norm.cdf(X,loc=mean,scale=std)
1-stats.norm.cdf(38,loc=34.422076,scale=9.131445)


# In[41]:


# P(MPG<40)
#stats.norm.cdf(X,loc=mean,scale=std)
stats.norm.cdf(40,loc=34.422076,scale=9.131445)


# In[42]:


# P (20<MPG<5)
#stats.norm.cdf(X,loc=mean,scale=std)-1-stats.norm.cdf(X,loc=mean,scale=std)
stats.norm.cdf(0.50,loc=34.422076,scale=9.131445)-stats.norm.cdf(0.20,loc=34.422076,scale=9.131445)


# In[43]:


sns.boxplot(df.MPG)     


# ### Q 20) Calculate probability from the given dataset for the below cases 
# 
#  
# 
# Data _set: Cars.csv 
# 
# ### Calculate the probability of MPG  of Cars for the below cases. 
# 
#        MPG <- Cars$MPG 
# 
# ### a) P(MPG>38) 
# 
# ### b) P(MPG<40) 
# 
# ### c) P (20<MPG<50) 

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


# In[45]:


cars=pd.read_csv("D:\Basic Statistics-1\Cars (3).csv")
cars


# In[46]:


sns.boxplot(cars.MPG)


# In[47]:


# P(MPG>38)
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())


# In[48]:


# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())


# In[49]:


# P (20<MPG<50)
stats.norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std())


# ### Q 21) Check whether the data follows normal distribution 
# 
# ### a) Check whether the MPG of Cars follows Normal Distribution  
# 
#         Dataset: Cars.csv 

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


cars=pd.read_csv("D:\Basic Statistics-1\Cars (3).csv")
cars


# In[53]:


sns.distplot(cars.MPG, label='Cars-MPG')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend();


# In[54]:


cars.MPG.mean()


# In[55]:


cars.MPG.median()


# ### b) Check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follows Normal Distribution  
# 
#        Dataset: wc-at.csv 

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


wcat=pd.read_csv("D:\Basic Statistics-1\wc-at (2).csv")
wcat


# In[58]:


# plotting distribution for Waist Circumference (Waist) 
sns.distplot(wcat.Waist)
plt.ylabel('density');


# In[59]:


# plotting distribution for Adipose Tissue (AT) 
sns.distplot(wcat.AT)
plt.ylabel('density');


# In[60]:


# WC
wcat.Waist.mean() , wcat.Waist.median()


# ### Q 22) Calculate the Z scores of 90% confidence interval,94% confidence
# ### interval, 60% confidence interval

# In[61]:


# Z-score of 90% confidence interval
stats.norm.ppf(0.95)


# In[62]:


# Z-score of 94% confidence interval
stats.norm.ppf(0.97)


# In[63]:


# Z-score of 60% confidence interval
stats.norm.ppf(0.8)


# ### Q 23) Calculate the t scores of 95% confidence interval, 96% confidence
# ### interval, 99% confidence interval for sample size of 25

# In[64]:


from scipy import stats
from scipy.stats import norm


# In[65]:


# t score of 95% confidence interval for sample size of 25
stats.t.ppf(0.975,24)  # df = n-1 = 24


# In[66]:


# t score of 96% confidence interval for sample size of 25
stats.t.ppf(0.98,24)


# In[67]:


# t scores of 99% confidence interval for sample size of 25
stats.t.ppf(0.995,24)


# ### Q 24) A Government company claims that an average light bulb lasts 270
# ### days. A researcher randomly selects 18 bulbs for testing. The sampled
# ### bulbs last an average of 260 days, with a standard deviation of 90 days. If
# ### the CEO&#39;s claim were true, what is the probability that 18 randomly
# ### selected bulbs would have an average life of no more than 260 days
# ### Hint:
# ### rcode  pt(tscore,df)
# ### df  degrees of freedom

# In[68]:


from scipy import stats
from scipy.stats import norm


# In[69]:


# Assume Null Hypothesis is: H0 = Avg life of Bulb >= 260 days
# Alternate hypothesis is: Ha = Avg life of bulb < 260 days


# In[70]:


# Find t-scores at x=260; t = (s_mean)-p_mean)/(s_SD/squrt(n))
t=(260-270)/(90/18**0.5)
t


# In[71]:


# Find P(x>=260) for null hypothesis


# In[72]:


# p_ value=1-stats.t.cdf(abs(t_scores),df=n-1)...Using cdf function
p_value = 1-stats.t.cdf(abs(-0.4714),df=17)
p_value


# In[73]:


# OR p_values=stats.t.sf(abs(t_score),df=n-1)...Using sf function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value


# ==============================================================================================================================
