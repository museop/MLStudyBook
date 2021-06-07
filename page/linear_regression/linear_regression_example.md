# Linear Regression 예제

## 간단한 Linear regression 예제

### 예제 데이터 만들기

먼저 $$y=3x + 5$$를 이루는 데이터를 만들어보자. 임의의 $$x$$ 값들을 만든 후, $$y$$ 값들을 식에 생성 해주면 된다.  이후 sklearn 패키지를 사용하기 위해 `reshape((-1, 1))`을 이용하여 $$x$$ 값을 (데이터의 수, 독립 변수의 수) 형태의 2차원 배열로 만든 것에 주의하자.

```python
import numpy as np
np.random.seed(47)

X = np.array(np.random.rand(50)).reshape((-1, 1)) # X.shape: (50, 1)
y = X * 3 + 5                                     # y.shape: (50, 1)
```

그리고 $$y$$ 값에 노이즈를 조금 추가해서 조금 더 현실적인 데이터를 만들어준다.

```python
noise = 0.2 * np.random.rand(50).reshape((-1, 1)) # noise
y += noise
```

데이터가 어떻게 생성되었는지 그려보자.

```python
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.show()
```

![linear_regression_exam1_data](img\linear_regression_exam1_data.PNG)



## 데이터 예측하기

앞서 생성한 데이터를 기반으로 선형모델을 만들고, 새로운 데이터를 넣어 예측이 잘 되는지 확인해보자. 사이킷럿 패키지를 이용하면 쉽게 linear regression을 수행할 수 있다.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```

선형 모델의 결정 계수는 다음과 같이 구한다.

```python
print("R^2 score:")
print(model.score(X, y)) # 0.9965439101104848
```

선형 모델의 기울기와 절편은 다음과 같이 구한다.

 ```python
print("Coef:")
print(model.coef_) # array([[2.9974268]])
print("Intercept:")
print(model.intercept_) # array([5.11253757])
 ```

위 기울기와 절편에 의하면 $$y=3x+5$$ 임을 알 수 있다. 이 식에 직접 새로운 데이터를 대입하여 값을 예측하여도 되고, 다음과 같이 `predict` 를 이용하여 여러 값들을 예측할 수 있다.

```python
print("Predict X=4:")
print(model.predict(np.array([[4]]))) # 4 * 3 + 5
```



## 전체 코드

```python
import numpy as np
np.random.seed(47)

X = np.array(np.random.rand(50)).reshape((-1, 1))
y = X * 3 + 5
noise = 0.2 * np.random.rand(50).reshape((-1, 1)) # noise
y += noise

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print("R^2 score:")
print(model.score(X, y))
print("Coef:")
print(model.coef_)
print("Intercept:")
print(model.intercept_)
print("Predict X=4:")
print(model.predict(np.array([[4]]))) # 4 * 3 + 5
```

