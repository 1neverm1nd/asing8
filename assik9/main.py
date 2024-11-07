import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Загрузка данных
df = pd.read_excel('xlls/price1.xlsx')

# Построение графика зависимости цены от площади
plt.scatter(df.area, df.price, color='red', marker='^')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Стоимость (млн.руб)')
plt.show()

# Создание и обучение модели линейной регрессии
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Прогнозирование
reg.predict(pd.DataFrame([[120]], columns=['area']))
reg.predict(df[['area']])

# Параметры модели
reg.coef_
reg.intercept_

# Построение графика с линией регрессии
plt.scatter(df.area, df.price, color='red', marker='^')
plt.xlabel('Площадь (кв.м.)')
plt.ylabel('Стоимость (млн.руб)')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

# Прогноз для новых данных
pred = pd.read_excel('xlls/prediction_price.xlsx')
pred['predicted prices'] = reg.predict(pred)

# Сохранение прогноза в новый Excel файл
pred.to_excel('new.xlsx', index=False)

# Работа с данными по ВВП России
df = pd.read_excel('xlls/gdprussia.xlsx')

# Построение графика зависимости ВВП от цены на нефть
plt.scatter(df.oilprice, df.gdp)
plt.xlabel('Цена на нефть (US$)')
plt.ylabel('ВВП России (млрд. US$)')
plt.show()

# Обучение модели для прогноза ВВП по цене на нефть
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

# Модель с двумя признаками (год и цена на нефть)
reg = linear_model.LinearRegression()
reg.fit(df[['year', 'oilprice']], df.gdp)

# Прогнозирование ВВП для новых значений
reg.predict(pd.DataFrame([[2025, 100]], columns=['year', 'oilprice']))

# Сохранение нового Excel файла с прогнозами
pred.to_excel('new.xlsx', index=False)
