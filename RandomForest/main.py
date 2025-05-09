import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Переменные для хранения лучших результатов
best_accuracy = 0
best_max_depth = None
best_max_leaf_nodes = None

# Обучение модели Random Forest с различными параметрами
for max_depth in range(1, 6):
    for max_leaf_nodes in range(2, 11):
        model = RandomForestClassifier(n_estimators=200, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, criterion='entropy', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценка ошибки классификации
        accuracy = accuracy_score(y_test, y_pred)
        error = 1 - accuracy

        if error < best_accuracy or best_accuracy == 0:
            best_accuracy = error
            best_max_depth = max_depth
            best_max_leaf_nodes = max_leaf_nodes

# Вывод результатов
print(f"Оптимальная max_depth: {best_max_depth}, max_leaf_nodes: {best_max_leaf_nodes}")
print(f"Средняя ошибка на тестовой выборке: {best_accuracy:.4f}")

# Визуализация результатов
# График реальных сортов ириса
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Реальные сорта ириса")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.subplot(1, 2, 2)
plt.title("Предсказанные сорта ириса")
y_pred = model.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()
