import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
url = 'https://gist.github.com/taejs/c4f9138e1a1bb376c989b5d175fe6e3f/raw/california_housing_train.csv'
df = pd.read_csv(url)

# Задание 1: Сравнение распределений пар числовых переменных
sns.pairplot(df)
plt.suptitle('Сравнение распределений пар числовых переменных', y=1.02)
plt.show()

# Задание 2: Построение тепловой карты корреляций
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Тепловая карта корреляций')
plt.show()

# Задание 3: Построение гистограмм для каждой переменной
df.hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Гистограммы распределения переменных', y=1.02)
plt.show()

# Задание 4: Нахождение переменной с наибольшей и наименьшей дисперсиями
variances = df.var()
max_variance = variances.idxmax(), variances.max()
min_variance = variances.idxmin(), variances.min()

print(f'Переменная с наибольшей дисперсией: {max_variance[0]} (дисперсия: {max_variance[1]:.4f})')
print(f'Переменная с наименьшей дисперсией: {min_variance[0]} (дисперсия: {min_variance[1]:.4f})')
