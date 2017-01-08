import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
preds = lm.predict(X_new)
data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.savefig('foo.png')