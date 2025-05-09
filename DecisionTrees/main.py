import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Загрузка датасета ириса
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Переменные для хранения наилучших параметров
best_depth = 0
best_leaf_nodes = 0
min_error = float('inf')

# Поиск оптимальных max_depth и max_leaf_nodes
for max_depth in range(1, 6):
    for max_leaf_nodes in range(2, 11):
        # Создание и обучение модели
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
        model.fit(X_train, y_train)

        # Предсказание классов на тестовой выборке
        y_pred = model.predict(X_test)

        # Вычисление ошибки классификации
        error = 1 - accuracy_score(y_test, y_pred)

        # Сохранение наилучших параметров
        if error < min_error:
            min_error = error
            best_depth = max_depth
            best_leaf_nodes = max_leaf_nodes

# Вывод результатов
print(f'Оптимальная глубина дерева (max_depth): {best_depth}')
print(f'Оптимальное количество листьев (max_leaf_nodes): {best_leaf_nodes}')
print(f'Минимальная ошибка на тестовой выборке: {min_error:.4f}')

# Обучение модели с оптимальными параметрами
optimal_model = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, max_leaf_nodes=best_leaf_nodes, random_state=42)
optimal_model.fit(X_train, y_train)

# Визуализация дерева решений
plt.figure(figsize=(12, 8))
plot_tree(optimal_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Дерево решений')
plt.show()

# Визуализация результатов
# График реальных сортов ирисов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='o', edgecolor='k')
plt.title('Реальные сорта ирисов')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# График предсказанных сортов
plt.subplot(1, 2, 2)
y_pred = optimal_model.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.title('Предсказанные сорта ирисов')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()
