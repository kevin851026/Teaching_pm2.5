import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#斜直線
# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = 3 * x - 5 + rng.randn(50)
# plt.scatter(x, y);
# # plt.show()
model = LinearRegression(fit_intercept=True)
# # print(x.shape)
# print(x[:, np.newaxis])
# print(y)
# model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# # plt.scatter(x, y)
# plt.plot(xfit, yfit);
# plt.show()

#多項式
poly_model = make_pipeline(PolynomialFeatures(5), LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
# X = 10 * rng.rand(100, 3)
# y = 0.5 + np.dot(X, [1.5, -1., 2.])
# print(X.shape)