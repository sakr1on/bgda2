import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Загрузка данных
# Загружаем датасет ирисов и используем только первые два признака для удобства визуализации.
iris = datasets.load_iris()
X = iris.data[:, :2]  # Используем только два признака для визуализации
y = iris.target  # Целевые метки (сорта ирисов)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели и поиск оптимального C
C_values = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]
best_C = C_values[0]  # Инициализация наилучшего значения C
best_accuracy = 0

for C in C_values:
    # Обучаем модель SVM с линейным ядром и текущим значением C
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)

    # Предсказание классов для тестовой выборки
    y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C

print(f'Оптимальное значение C: {best_C}, Средняя ошибка: {1 - best_accuracy}')

# Визуализация результатов
# Создаем сетку для графика, чтобы отобразить разделяющие поверхности.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Обучаем модель с оптимальным C для визуализации
svm_best = SVC(kernel='linear', C=best_C, random_state=42)
svm_best.fit(X_train, y_train)

# Предсказание для сетки, чтобы визуализировать разделяющую поверхность
Z = svm_best.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# График 1: Реальные сорта ирисов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
# Визуализируем реальные классы на тестовой выборке
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='o', edgecolor='k')
plt.title('Реальные сорта ирисов')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  # Отображаем разделяющую поверхность

# График 2: Предсказанные сорта ирисов
plt.subplot(1, 2, 2)
# Визуализируем предсказанные классы на тестовой выборке
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.title('Предсказанные сорта ирисов')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')  # Отображаем разделяющую поверхность

# Настройка графиков и отображение
plt.tight_layout()
plt.show()
