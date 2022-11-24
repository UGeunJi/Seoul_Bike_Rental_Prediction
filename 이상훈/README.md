
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
```

```python
train = pd.read_csv('data/train.csv') 
test = pd.read_csv('data/test.csv')
```

```python
# 한글폰트 출력하기 위함
mpl.rc('font', family='Malgun Gothic')
```

# train 정보 파악하기

```python
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>20</td>
      <td>16.3</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>89.0</td>
      <td>576.0</td>
      <td>0.027</td>
      <td>76.0</td>
      <td>33.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>13</td>
      <td>20.1</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>48.0</td>
      <td>916.0</td>
      <td>0.042</td>
      <td>73.0</td>
      <td>40.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>6</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>79.0</td>
      <td>1382.0</td>
      <td>0.033</td>
      <td>32.0</td>
      <td>19.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>23</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>54.0</td>
      <td>946.0</td>
      <td>0.040</td>
      <td>75.0</td>
      <td>64.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>18</td>
      <td>29.5</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>7.0</td>
      <td>2000.0</td>
      <td>0.057</td>
      <td>27.0</td>
      <td>11.0</td>
      <td>431.0</td>
    </tr>
  </tbody>
</table>
</div>

- id 고유 id
- hour 시간
- temperature 기온
- precipitation 비가 오지 않았으면 0, 비가 오면 1
- windspeed 풍속(평균)
- humidity 습도
- visibility 시정(視程), 시계(視界)(특정 기상 상태에 따른 가시성을 의미)
- ozone 오존
- pm10 미세먼지(머리카락 굵기의 1/5에서 1/7 크기의 미세먼지)
- pm2.5 미세먼지(머리카락 굵기의 1/20에서 1/30 크기의 미세먼지)
- count 시간에 따른 따릉이 대여 수

```python
train.info() # missing valuer값이 존재하는 것을 볼 수 있다
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 11 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   id                      1459 non-null   int64  
 1   hour                    1459 non-null   int64  
 2   hour_bef_temperature    1457 non-null   float64
 3   hour_bef_precipitation  1457 non-null   float64
 4   hour_bef_windspeed      1450 non-null   float64
 5   hour_bef_humidity       1457 non-null   float64
 6   hour_bef_visibility     1457 non-null   float64
 7   hour_bef_ozone          1383 non-null   float64
 8   hour_bef_pm10           1369 non-null   float64
 9   hour_bef_pm2.5          1342 non-null   float64
 10  count                   1459 non-null   float64
dtypes: float64(9), int64(2)
memory usage: 125.5 KB
```

```python
train.isnull().sum() # 항목별 missing value
```

```
id                          0
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
dtype: int64
```

# 상관계수 파악하기

```python
corr = train.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt ='.2f', cmap='Reds', linewidth=0.5)
```

```
<AxesSubplot:>
```

# 연속형 데이터 시각화

- 산점도와 회귀선 분석

```python
plt.title("시간 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='count', y='hour')
```

```
<AxesSubplot:title={'center':'시간 별 자전거 렌탈 수'}, xlabel='count', ylabel='hour'>
```

**위와 같이 범주형 형태로 나오는 이유는 hour(시간)이 
0~23시로 정해져 있기때문에 범주형으로 볼 수 있다**

```python
plt.title("기온 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='count', y='hour_bef_temperature')
sns.lmplot(data=train, x='count', y='hour_bef_temperature')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a4d3c0a00>
```

```python
plt.title("습도 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='count', y='hour_bef_humidity')
sns.lmplot(data=train, x='count', y='hour_bef_humidity')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a4d3974c0>
```

```python
plt.title("풍속 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='count', y='hour_bef_windspeed')
sns.lmplot(data=train, x='count', y='hour_bef_windspeed')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a5d166b50>
```

```python
plt.title("오존 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='count', y='hour_bef_ozone')
sns.lmplot(data=train, x='count', y='hour_bef_ozone')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a5d239940>
```

```python
plt.title("습도에 따른 기상상태(가시성)")
sns.scatterplot(data=train, x='hour_bef_visibility', y='hour_bef_humidity')
sns.lmplot(data=train, x='hour_bef_visibility', y='hour_bef_humidity')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a44563f40>
```

```python
plt.title("초미세먼지에 따른 기상상태(가시성)")
sns.scatterplot(data=train, x='hour_bef_visibility', y='hour_bef_pm2.5')
sns.lmplot(data=train, x='hour_bef_visibility', y='hour_bef_pm2.5')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a455613a0>
```

```python
plt.title("습도에 따른 온도")
sns.scatterplot(data=train, x='hour_bef_temperature', y='hour_bef_humidity')
sns.lmplot(data=train, x='hour_bef_temperature', y='hour_bef_humidity')
```

```
<seaborn.axisgrid.FacetGrid at 0x15a4454b430>
```

```python
plt.title("습도, 온도상태 별 자전거 렌탈 수")
sns.scatterplot(data=train, x='hour_bef_temperature', y='hour_bef_humidity', hue='count')
```

```
<AxesSubplot:title={'center':'습도, 온도상태 별 자전거 렌탈 수'}, xlabel='hour_bef_temperature', ylabel='hour_bef_humidity'>
```

## 모든 연속형 변수의 산점도와 데이터 분포를 나타냄

```python
sns.pairplot(train)
```

```
<seaborn.axisgrid.PairGrid at 0x15a4eab9850>
```

**선택적으로 연속형 변수 데이터를 뽑아옴**

- 렌탈 수, 기온, 풍속, 습도, (시간)

```python
sns.pairplot(train[['count','hour_bef_temperature','hour_bef_windspeed','hour_bef_humidity','hour']],hue='hour')
```

```
<seaborn.axisgrid.PairGrid at 0x15a59dc8b20>
```
