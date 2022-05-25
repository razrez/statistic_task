# -*- coding: utf-8 -*-
import math
from collections import Counter
from scipy import stats  # модуль статистических функций
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def load_data():
    """Загрузка всего .csv файла"""
    return pd.read_csv('winequality-red.csv',
                       sep=';', encoding='UTF-8')


def exec_columns_names():
    """Получить имена полей кадра данных"""
    return load_data().columns


def exec_column(column_name):
    """Получить кадр данных по названию поля"""
    return load_data()[column_name]


def mean(xs):
    """Среднее значение числового ряда"""
    return sum(xs) / len(xs)


def median(xs):
    """Медиана числового ряда"""
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return sorted(xs)[mid]
    else:
        return mean(sorted(xs)[mid - 1:][:2])


def variance(xs):
    """Дисперсия (варианс) числового ряда,
       несмещенная дисперсия при n <= 30"""
    mu = mean(xs)
    n = len(xs)
    n = n - 1 if n in range(1, 30) else n
    square_deviation = lambda x: (x - mu) ** 2
    return sum(map(square_deviation, xs)) / n


def standard_deviation(xs):
    """Стандартное отклонение числового ряда"""
    return math.sqrt(variance(xs))


def saveplot(path):
    plt.tight_layout()
    plt.savefig('gists/' + path)


def built_hist(column):
    """Построить гистограмму частотных интервалов
        с  1 + 3.322 * math.log10(len(column)) интервалами"""
    m = int(abs(1 + 3.322 * math.log10(len(column))))
    column.hist(bins=m, color = "purple")
    plt.xlabel('Процент Алкоголя')
    plt.ylabel('Частота')
    # saveplot(f'built_hist_optimal{random.Random(100000)}.png')
    plt.show()


def m_k(column, k):
    """Центральный момент k-ого порядка"""
    x_av = mean(column)
    cr = Counter(column)
    mk = 0
    for val in cr:
        ni = cr[val]
        mk += math.pow((val - x_av), k) * ni
    mk /= len(column)
    return mk


def a_3(column):
    """Ассиметрия характеризует меру скошенности графика влево или вправо
        если a_3: < 0.25 => незначительная; > 0.5 => существенная"""
    return m_k(column, 3) / standard_deviation(column) ** 3


def excess(column):
    """ Если > 0, то эмпирическое распределение является
        более высоким («островершинным») – относительно «эталонного» нормального распределения.

        Если же < 0  – то более низким и пологим. И чем больше  по модулю, тем «аномальнее»
        высота в ту или иную сторону."""

    m4 = m_k(column, 4)
    sd4 = standard_deviation(column) ** 4
    return (m4 / sd4) - 3


def trust_interval_x(column, alpha):
    """ Доверительный интервал для генерального среднего по Стьюденту.
      Если нужно найти только полуширину доверительного интервала - ту величину,
     которая ставится после знака ±, то она вычисляется так:"""

    x = mean(column)
    a_err = standard_deviation(column)
    n = len(column)
    plus_minus = abs(stats.t.ppf(alpha / 2, n - 2)) * a_err
    print(f"Доверительный интервал для генерального среднего по Стьюденту надежности "
          f"{alpha * 100}%: [{x - plus_minus} ; {x + plus_minus}]")


def trust_interval_chi(column, gamma):
    """ Доверительный интервал для дисперсии нормальной случайной величины
        при неизвестном математическом ожидании
        !!тут домножается на выборочную дисперсию(слева/справа)
            => так и оценивается уже генеральная дисперсия"""
    # имитирую выборку
    part = column[:1200]
    a_err = variance(part)
    n = len(part)
    k = n - 1

    alpha1 = (1 - gamma) / 2
    alpha2 = (1 + gamma) / 2
    chi1_2 = stats.chi2.ppf(alpha1, k)
    chi2_2 = stats.chi2.ppf(alpha2, k)

    end = ((n - 1) * a_err) / chi1_2
    start = ((n - 1) * a_err) / chi2_2
    print(f"Доверительный интервал для генеральной дисперсии надежности(гамма?) "
          f"{gamma * 100}%: [{start} ; {end}]")


def correlation(x, y):
    corr_result = np.corrcoef(x, y)
    print("Коэффициент корреляции:", corr_result)


def show_correlation():
    dataset = pd.read_csv('winequality-red.csv',
                          sep=';', encoding='UTF-8')
    x = dataset['free sulfur dioxide']
    y = dataset['total sulfur dioxide']
    data = {'x': x, 'y': y}
    df = pd.DataFrame(data)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    regressor = LinearRegression()
    regressor.fit(x, y)
    b = regressor.intercept_
    a = regressor.coef_
    print(f"y = {a.mean()}x + {b}")

    # Корреляция
    # Прогноз x_max + 2
    print("predict for y(x_max + 2): ", regressor.predict([[df.x.max() + 2]]))
    plt.scatter(x, y, color='orange')
    plt.plot(x, regressor.predict(x), color='red')
    plt.title('Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    firstData = exec_column('alcohol')

    print('среднее:', mean(firstData))
    print("медиана библиотечная:", firstData.median())
    print('медиана моя:', median(firstData))
    print('Дисперсия моя:', variance(firstData))
    print('Стандартное отклонение мое:', standard_deviation(firstData))
    print('Стандартное отклонение библиотечное:', firstData.std(ddof=0))
    print("Мода:", (stats.mode(firstData)[0][0]))
    built_hist(firstData)
    print("Ассиметрия:", a_3(firstData))
    print("Коэффициент эксцесса:", excess(firstData))

    trust_interval_x(firstData, 0.95)
    trust_interval_chi(firstData, 0.91)

    secondData = exec_column('sulphates')
    correlation(firstData, secondData)
    show_correlation()
