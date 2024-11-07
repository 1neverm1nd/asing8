import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Загрузка данных
df = pd.read_excel('xlls/gdprussia.xlsx')

# Построение графика зависимости ВВП от цены на нефть
plt.scatter(df.oilprice, df.gdp)
plt.xlabel('Цена на нефть (US$)')
plt.ylabel('ВВП России (млрд. US$)')
plt.show()

# Создание и обучение модели линейной регрессии по цене на нефть
reg = linear_model.LinearRegression()
reg.fit(df[['oilprice']], df.gdp)

# Построение графика с линией регрессии
plt.scatter(df.oilprice, df.gdp)
plt.xlabel('Цена на нефть (US$)')
plt.ylabel('ВВП России (млрд. US$)')
plt.plot(df.oilprice, reg.predict(df[['oilprice']]), color='blue')
plt.show()

# Прогноз ВВП при цене нефти 150$
reg.predict(pd.DataFrame([[150]], columns=['oilprice']))

# Создание и обучение модели линейной регрессии по году и цене на нефть
reg = linear_model.LinearRegression()
reg.fit(df[['year', 'oilprice']], df.gdp)

# Прогнозирование ВВП по году и цене на нефть
reg.predict(df[['year', 'oilprice']])
reg.predict(pd.DataFrame([[2025, 100]], columns=['year', 'oilprice']))
