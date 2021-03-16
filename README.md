# titanic
泰坦尼克号数据练习


目录
1. 提出问题（Business Understanding ）
2. 理解数据（Data Understanding）
 * 采集数据
 * 导入数据
 * 查看数据集信息
3. 数据清洗（Data Preparation ）
 * 数据预处理
 * 特征工程（Feature Engineering）
4. 构建模型（Modeling） 
5. 模型评估（Evaluation） 
6. 方案实施 （Deployment）
 * 提交结果到Kaggle
 * 报告撰写

## 1.提出问题？

什么样的人在泰坦尼克号更容易存活？

## 2.理解数据

### 2.1采集数据

数据来源：kaggle泰坦尼克号项目：https://www.kaggle.com/c/titanic

### 2.2 导入数据


```python
#忽略警告提示
import warnings
warnings.filterwarnings('ignore')
#导入数据处理包
import numpy as np
import pandas as pd
```


```python
#导入数据
#导入训练数据
train=pd.read_csv('train.csv')
#导入测试数据
test=pd.read_csv('test.csv')
print('训练数据集：',train.shape,'测试数据集:',test.shape)
```

    训练数据集： (891, 12) 测试数据集: (418, 11)
    


```python
#查看train 数据集信息
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看test 数据集信息
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('训练数据集行数：',rowNum_train,'测试数据集行数：',rowNum_test)
```

    训练数据集行数： 891 测试数据集行数： 418
    


```python
#合并数据集方便对两个数据进行同时清洗
full=train.append(test,ignore_index=True)
print('合并后的数据集',full.shape)
```

    合并后的数据集 (1309, 12)
    

### 2.3查看数据集信息


```python
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
describe只能查看数据类型的描述统计信息，对于其他类型的数据不显示，比如字符串类型姓名（name），客舱号（Cabin）
这很好理解，因为描述统计指标是计算数值，所以需要该列的数据类型是数据
'''
full.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1046.000000</td>
      <td>1308.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>33.295479</td>
      <td>0.385027</td>
      <td>655.000000</td>
      <td>2.294882</td>
      <td>0.498854</td>
      <td>0.383838</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.413493</td>
      <td>51.758668</td>
      <td>0.865560</td>
      <td>378.020061</td>
      <td>0.837836</td>
      <td>1.041658</td>
      <td>0.486592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>7.895800</td>
      <td>0.000000</td>
      <td>328.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>14.454200</td>
      <td>0.000000</td>
      <td>655.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>39.000000</td>
      <td>31.275000</td>
      <td>0.000000</td>
      <td>982.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>9.000000</td>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看每一列的数据类型和指标
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1046 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1308 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

## 3.数据清洗

### 3.1数据预处理

### 缺失值处理

在前面，理解数据阶段，我们发现数据总共有1309行。

其中数据类型列：年龄（Age）、船舱号（Cabin）里面有缺失数据：
1）年龄（Age）里面数据总数是1046条，缺失了1309-1046=263，缺失率263/1309=20%
2）船票价格（Fare）里面数据总数是1308条，缺失了1条数据

字符串列：
1）登船港口（Embarked）里面数据总数是1307，只缺失了2条数据，缺失比较少
2）船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失比较大
这为下一步数据清洗指明了方向，知道哪些数据缺失数据，才能有针对性的处理。

很多机器学习算法为了训练模型，要求所传入的特征中不能有空值。


1. 如果是数值类型，用平均值取代
2. 如果是分类数据，用最常见的类别取代
3. 使用模型预测缺失值，例如：K-NN


```python
#对于数据类型，处理缺失值最简单的方法就是用平均数来填充缺失值
print('处理前：')
full.info()
```

    处理前：
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1046 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1308 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    


```python
#年龄（Age)
full['Age']=full['Age'].fillna(full['Age'].mean())
#船票价格（Fare)
full['Fare']=full['Fare'].fillna(full['Fare'].mean())
print('处理后：')
full.info()
```

    处理后：
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1309 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1309 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    


```python
#检查数据是否正常
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
总数据是1309
字符串列：
1）登船港口（Embarked）里面数据总数是1307，只缺失了2条数据，缺失比较少
2）船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，缺失比较大
'''
#登船港口（Embarked）：查看内部数据
'''
出发地点：S=英国南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
full['Embarked'].head()
```




    0    S
    1    C
    2    S
    3    S
    4    S
    Name: Embarked, dtype: object




```python
#分类变量Embarked,看下最常见的类别，用其填充
```


```python
full['Embarked'].value_counts()
```




    S    914
    C    270
    Q    123
    Name: Embarked, dtype: int64




```python
#S类别最常见，我们将缺失值填充为最频繁出现的值：出发地点：S=英国南安普顿Southampton
full['Embarked']=full['Embarked'].fillna('S')
```


```python
#船舱号(Cabin):查看内部数据
full['Cabin'].head()
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: Cabin, dtype: object




```python
#缺失数据比较多，船舱号（Cabin)缺失值为U，表示未知（Uknown)
full['Cabin']=full['Cabin'].fillna('U')
```


```python
#检查数据是否正常
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看最终缺失值处理情况，记住生成情况（Survived）这里一列是我们的标签，用来做机器学习预测的，不需要处理这一列
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1309 non-null float64
    Cabin          1309 non-null object
    Embarked       1309 non-null object
    Fare           1309 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

### 3.2特征提取

### 3.2.1 数据分类

查看数据类型，分为三种数据类型，并对类别数据处理：用数值代替类别，并进行One-hot编码

1.数值类型：

乘客编号（PassengerId），年龄（Age），船票价格（Fare），同代直系亲属人数（SibSp），不同代直系亲属人数（Parch）

2.时间序列：无

3.分类数据：

1）有直接类别的

乘客性别（Sex）：男性male，女性female

登船港口（Embarked）：出发地点S=英国南安普顿Southampton，途径地点1：C=法国 瑟堡市Cherbourg，途径地点2：Q=爱尔兰 昆士敦Queenstown

客舱等级（Pclass）：1=1等舱，2=2等舱，3=3等舱

2）字符串类型：可能从这里面提取出特征来，也归到分类数据中

乘客姓名（Name）

客舱号（Cabin）

船票编号（Ticket）


```python
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1309 non-null float64
    Cabin          1309 non-null object
    Embarked       1309 non-null object
    Fare           1309 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

###  3.2.1分类数据：有直接类别的

### 性别



```python
#查看性别这列数据
full['Sex'].head()
```




    0      male
    1    female
    2    female
    3    female
    4      male
    Name: Sex, dtype: object




```python
'''
将性别映射为数值
male对应数值1，female对应数值0
'''
sex_mapDic={'male':1,
           'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDic)
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>



### 登船港口


```python
'''
登船港口(Embarked)的值是：
出发地点：S=英国南安普顿Southampton
途径地点1：C=法国 瑟堡市Cherbourg
途径地点2：Q=爱尔兰 昆士敦Queenstown
'''
#查看该类数据内容
full['Embarked'].head()
```




    0    S
    1    C
    2    S
    3    S
    4    S
    Name: Embarked, dtype: object




```python
#存放提取后的特征
embarkedDf=pd.DataFrame()
'''
使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables），列名前缀是Embarked
'''
embarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked')
embarkedDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,embarkedDf],axis=1)
'''
因为已经使用登船港口(Embarked)进行了one-hot编码产生了它的虚拟变量（dummy variables）
所以这里把登船港口(Embarked)删掉
'''

'''
上面drop删除某一列代码解释：
因为drop(name,axis=1)里面指定了name是哪一列，比如指定的是A这一列，axis=1表示按行操作。
那么结合起来就是把A列里面每一行删除，最终结果是删除了A这一列.
简单来说，使用drop删除某几列的方法记住这个语法就可以了：drop([列名1,列名2],axis=1)
'''
full.drop('Embarked',axis=1,inplace=True)
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 客舱等级


```python
'''
客舱等级(Pclass):
1=1等舱，2=2等舱，3=3等舱
'''
pclassDf=pd.DataFrame()
pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
pclassDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#添加 one-hot 编码产生的虚拟变量（dummy variablese）到泰坦尼克数据集full
full=pd.concat([full,pclassDf],axis=1)
#删掉客舱等级这列
full.drop('Pclass',axis=1,inplace=True)
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.1分类数据：字符串类型

字符串类型：可能从这里面提取出特征来，也归到分类数据中，这里数据有：

乘客姓名（Name）
客舱号（Cabin）
船票编号（Ticket）

### 从姓名中提取头衔


```python
'''
查看姓名这一列长啥样
注意到在乘客名字（Name）中，有一个非常显著的特点：
乘客头衔每个名字当中都包含了具体的称谓或者说是头衔，将这部分信息提取出来后可以作为非常有用一个新变量，可以帮助我们进行预测。
例如：
Braund, Mr. Owen Harris
Heikkinen, Miss. Laina
Oliva y Ocana, Dona. Fermina
Peter, Master. Michael J
'''
full[ 'Name' ].head()
```




    0                              Braund, Mr. Owen Harris
    1    Cumings, Mrs. John Bradley (Florence Briggs Th...
    2                               Heikkinen, Miss. Laina
    3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    4                             Allen, Mr. William Henry
    Name: Name, dtype: object




```python
#练习从字符串中提取头衔，例如Mr
#split用于字符串分割，返回一个列表
#我们看到姓名中'Braund, Mr. Owen Harris'，逗号前面的是“名”，逗号后面是‘头衔. 姓’
name1='Braund, Mr. Owen Harris'
'''
split用于字符串按分隔符分割，返回一个列表。这里按逗号分隔字符串
也就是字符串'Braund, Mr. Owen Harris'被按分隔符,'拆分成两部分[Braund,Mr. Owen Harris]
你可以把返回的列表打印出来瞧瞧，这里获取到列表中元素序号为1的元素，也就是获取到头衔所在的那部分，即Mr. Owen Harris这部分
'''
#Mr. Owen Harris
str1=name1.split( ',' )[1] 
'''
继续对字符串Mr. Owen Harris按分隔符'.'拆分，得到这样一个列表[Mr, Owen Harris]
这里获取到列表中元素序号为0的元素，也就是获取到头衔所在的那部分Mr
'''
#Mr.
str2=str1.split( '.' )[0]
#strip() 方法用于移除字符串头尾指定的字符（默认为空格）
str3=str2.strip()
str3
```




    'Mr'




```python
'''
定义函数：从姓名中获取头衔
'''
def getTitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3
```


```python
#存放提取后的特征
titleDf=pd.DataFrame()
#map函数： 会根据提供的函数对指定序列做映射。
titleDf['Title']=full['Name'].map(getTitle)
titleDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
titleDf['Title'].value_counts()
```




    Mr              757
    Miss            260
    Mrs             197
    Master           61
    Rev               8
    Dr                8
    Col               4
    Ms                2
    Mlle              2
    Major             2
    Sir               1
    Capt              1
    Don               1
    Dona              1
    Lady              1
    Jonkheer          1
    the Countess      1
    Mme               1
    Name: Title, dtype: int64




```python
'''
定义以下几种头衔类别：
Officer政府官员
Royalty王室（皇室）
Mr已婚男士
Mrs已婚妇女
Miss年轻未婚女子
Master有技能的人/教师
'''
#姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
titleDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#添加 one-hot 编码产生的虚拟变量（dummy variablese）到泰坦尼克数据集full
full=pd.concat([full,titleDf],axis=1)
#删除姓名这一列
full.drop('Name',axis=1,inplace=True)
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>...</th>
      <th>Embarked_S</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### 从客舱号中提取客舱类别


```python
#补充知识：匿名函数
'''
python 使用 lambda 来创建匿名函数。
所谓匿名，意即不再使用 def 语句这样标准的形式定义一个函数，如下：
lambda 参数1，参数2：函数体或者表达式
'''
sum=lambda a,b:a+b
#调用sum函数
print('相加后的值为：',sum(10,20))
```

    相加后的值为： 30
    


```python
'''
客舱号的首字母是客舱类别
'''
#查看客舱号数据
full['Cabin'].head()
```




    0       U
    1     C85
    2       U
    3    C123
    4       U
    Name: Cabin, dtype: object




```python
#存放客舱数据
cabinDf=pd.DataFrame()
'''
客场号的类别值是首字母，例如：
C85 类别映射为首字母C
'''
full['Cabin']=full['Cabin'].map(lambda c:c[0])
##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')
cabinDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full=pd.concat([full,cabinDf],axis=1)
#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)
full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>
      <th>Royalty</th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 29 columns):
    Age            1309 non-null float64
    Fare           1309 non-null float64
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Sex            1309 non-null int64
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    Embarked_C     1309 non-null uint8
    Embarked_Q     1309 non-null uint8
    Embarked_S     1309 non-null uint8
    Pclass_1       1309 non-null uint8
    Pclass_2       1309 non-null uint8
    Pclass_3       1309 non-null uint8
    Master         1309 non-null uint8
    Miss           1309 non-null uint8
    Mr             1309 non-null uint8
    Mrs            1309 non-null uint8
    Officer        1309 non-null uint8
    Royalty        1309 non-null uint8
    Cabin_A        1309 non-null uint8
    Cabin_B        1309 non-null uint8
    Cabin_C        1309 non-null uint8
    Cabin_D        1309 non-null uint8
    Cabin_E        1309 non-null uint8
    Cabin_F        1309 non-null uint8
    Cabin_G        1309 non-null uint8
    Cabin_T        1309 non-null uint8
    Cabin_U        1309 non-null uint8
    dtypes: float64(3), int64(4), object(1), uint8(21)
    memory usage: 108.7+ KB
    

### 建立家庭人数和家庭号码


```python
#存放家庭信息
familyDf=pd.DataFrame()
'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize']=full['Parch']+full['SibSp']+1

'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2>=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''
#if 条件为真的时候返回if前面内容，否则返回0
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s: 1 if s==1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s: 1 if 2<= s<=4 else 0)
familyDf['Family_Large']=familyDf['FamilySize'].map(lambda s: 1 if s>=5 else 0)
familyDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FamilySize</th>
      <th>Family_Single</th>
      <th>Family_Small</th>
      <th>Family_Large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full=pd.concat([full,familyDf],axis=1)
full.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>FamilySize</th>
      <th>Family_Single</th>
      <th>Family_Small</th>
      <th>Family_Large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
full.shape
```




    (1309, 33)



### 3.3 特征选择


```python
#相关系数法计算各个特征的相关系数
corrDf=full.corr()
corrDf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>FamilySize</th>
      <th>Family_Single</th>
      <th>Family_Small</th>
      <th>Family_Large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.171521</td>
      <td>-0.130872</td>
      <td>0.025731</td>
      <td>0.057397</td>
      <td>-0.190747</td>
      <td>-0.070323</td>
      <td>0.076179</td>
      <td>-0.012718</td>
      <td>-0.059153</td>
      <td>...</td>
      <td>0.132886</td>
      <td>0.106600</td>
      <td>-0.072644</td>
      <td>-0.085977</td>
      <td>0.032461</td>
      <td>-0.271918</td>
      <td>-0.196996</td>
      <td>0.116675</td>
      <td>-0.038189</td>
      <td>-0.161210</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.171521</td>
      <td>1.000000</td>
      <td>0.221522</td>
      <td>0.031416</td>
      <td>-0.185484</td>
      <td>0.160224</td>
      <td>0.257307</td>
      <td>0.286241</td>
      <td>-0.130054</td>
      <td>-0.169894</td>
      <td>...</td>
      <td>0.072737</td>
      <td>0.073949</td>
      <td>-0.037567</td>
      <td>-0.022857</td>
      <td>0.001179</td>
      <td>-0.507197</td>
      <td>0.226465</td>
      <td>-0.274826</td>
      <td>0.197281</td>
      <td>0.170853</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.130872</td>
      <td>0.221522</td>
      <td>1.000000</td>
      <td>0.008942</td>
      <td>-0.213125</td>
      <td>0.373587</td>
      <td>0.081629</td>
      <td>-0.008635</td>
      <td>-0.100943</td>
      <td>0.071881</td>
      <td>...</td>
      <td>-0.027385</td>
      <td>0.001084</td>
      <td>0.020481</td>
      <td>0.058325</td>
      <td>-0.012304</td>
      <td>-0.036806</td>
      <td>0.792296</td>
      <td>-0.549022</td>
      <td>0.248532</td>
      <td>0.624627</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>0.025731</td>
      <td>0.031416</td>
      <td>0.008942</td>
      <td>1.000000</td>
      <td>0.013406</td>
      <td>-0.055224</td>
      <td>-0.005007</td>
      <td>0.048101</td>
      <td>0.011585</td>
      <td>-0.049836</td>
      <td>...</td>
      <td>0.000549</td>
      <td>-0.008136</td>
      <td>0.000306</td>
      <td>-0.045949</td>
      <td>-0.023049</td>
      <td>0.000208</td>
      <td>-0.031437</td>
      <td>0.028546</td>
      <td>0.002975</td>
      <td>-0.063415</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.057397</td>
      <td>-0.185484</td>
      <td>-0.213125</td>
      <td>0.013406</td>
      <td>1.000000</td>
      <td>-0.109609</td>
      <td>-0.543351</td>
      <td>-0.066564</td>
      <td>-0.088651</td>
      <td>0.115193</td>
      <td>...</td>
      <td>-0.057396</td>
      <td>-0.040340</td>
      <td>-0.006655</td>
      <td>-0.083285</td>
      <td>0.020558</td>
      <td>0.137396</td>
      <td>-0.188583</td>
      <td>0.284537</td>
      <td>-0.255196</td>
      <td>-0.077748</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.190747</td>
      <td>0.160224</td>
      <td>0.373587</td>
      <td>-0.055224</td>
      <td>-0.109609</td>
      <td>1.000000</td>
      <td>-0.035322</td>
      <td>-0.048396</td>
      <td>-0.048678</td>
      <td>0.073709</td>
      <td>...</td>
      <td>-0.015727</td>
      <td>-0.027180</td>
      <td>-0.008619</td>
      <td>0.006015</td>
      <td>-0.013247</td>
      <td>0.009064</td>
      <td>0.861952</td>
      <td>-0.591077</td>
      <td>0.253590</td>
      <td>0.699681</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.070323</td>
      <td>0.257307</td>
      <td>0.081629</td>
      <td>-0.005007</td>
      <td>-0.543351</td>
      <td>-0.035322</td>
      <td>1.000000</td>
      <td>0.168240</td>
      <td>0.003650</td>
      <td>-0.149683</td>
      <td>...</td>
      <td>0.150716</td>
      <td>0.145321</td>
      <td>0.057935</td>
      <td>0.016040</td>
      <td>-0.026456</td>
      <td>-0.316912</td>
      <td>0.016639</td>
      <td>-0.203367</td>
      <td>0.279855</td>
      <td>-0.125147</td>
    </tr>
    <tr>
      <th>Embarked_C</th>
      <td>0.076179</td>
      <td>0.286241</td>
      <td>-0.008635</td>
      <td>0.048101</td>
      <td>-0.066564</td>
      <td>-0.048396</td>
      <td>0.168240</td>
      <td>1.000000</td>
      <td>-0.164166</td>
      <td>-0.778262</td>
      <td>...</td>
      <td>0.107782</td>
      <td>0.027566</td>
      <td>-0.020010</td>
      <td>-0.031566</td>
      <td>-0.014095</td>
      <td>-0.258257</td>
      <td>-0.036553</td>
      <td>-0.107874</td>
      <td>0.159594</td>
      <td>-0.092825</td>
    </tr>
    <tr>
      <th>Embarked_Q</th>
      <td>-0.012718</td>
      <td>-0.130054</td>
      <td>-0.100943</td>
      <td>0.011585</td>
      <td>-0.088651</td>
      <td>-0.048678</td>
      <td>0.003650</td>
      <td>-0.164166</td>
      <td>1.000000</td>
      <td>-0.491656</td>
      <td>...</td>
      <td>-0.061459</td>
      <td>-0.042877</td>
      <td>-0.020282</td>
      <td>-0.019941</td>
      <td>-0.008904</td>
      <td>0.142369</td>
      <td>-0.087190</td>
      <td>0.127214</td>
      <td>-0.122491</td>
      <td>-0.018423</td>
    </tr>
    <tr>
      <th>Embarked_S</th>
      <td>-0.059153</td>
      <td>-0.169894</td>
      <td>0.071881</td>
      <td>-0.049836</td>
      <td>0.115193</td>
      <td>0.073709</td>
      <td>-0.149683</td>
      <td>-0.778262</td>
      <td>-0.491656</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.056023</td>
      <td>0.002960</td>
      <td>0.030575</td>
      <td>0.040560</td>
      <td>0.018111</td>
      <td>0.137351</td>
      <td>0.087771</td>
      <td>0.014246</td>
      <td>-0.062909</td>
      <td>0.093671</td>
    </tr>
    <tr>
      <th>Pclass_1</th>
      <td>0.362587</td>
      <td>0.599956</td>
      <td>-0.013033</td>
      <td>0.026495</td>
      <td>-0.107371</td>
      <td>-0.034256</td>
      <td>0.285904</td>
      <td>0.325722</td>
      <td>-0.166101</td>
      <td>-0.181800</td>
      <td>...</td>
      <td>0.275698</td>
      <td>0.242963</td>
      <td>-0.073083</td>
      <td>-0.035441</td>
      <td>0.048310</td>
      <td>-0.776987</td>
      <td>-0.029656</td>
      <td>-0.126551</td>
      <td>0.165965</td>
      <td>-0.067523</td>
    </tr>
    <tr>
      <th>Pclass_2</th>
      <td>-0.014193</td>
      <td>-0.121372</td>
      <td>-0.010057</td>
      <td>0.022714</td>
      <td>-0.028862</td>
      <td>-0.052419</td>
      <td>0.093349</td>
      <td>-0.134675</td>
      <td>-0.121973</td>
      <td>0.196532</td>
      <td>...</td>
      <td>-0.037929</td>
      <td>-0.050210</td>
      <td>0.127371</td>
      <td>-0.032081</td>
      <td>-0.014325</td>
      <td>0.176485</td>
      <td>-0.039976</td>
      <td>-0.035075</td>
      <td>0.097270</td>
      <td>-0.118495</td>
    </tr>
    <tr>
      <th>Pclass_3</th>
      <td>-0.302093</td>
      <td>-0.419616</td>
      <td>0.019521</td>
      <td>-0.041544</td>
      <td>0.116562</td>
      <td>0.072610</td>
      <td>-0.322308</td>
      <td>-0.171430</td>
      <td>0.243706</td>
      <td>-0.003805</td>
      <td>...</td>
      <td>-0.207455</td>
      <td>-0.169063</td>
      <td>-0.041178</td>
      <td>0.056964</td>
      <td>-0.030057</td>
      <td>0.527614</td>
      <td>0.058430</td>
      <td>0.138250</td>
      <td>-0.223338</td>
      <td>0.155560</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>-0.363923</td>
      <td>0.011596</td>
      <td>0.253482</td>
      <td>0.002254</td>
      <td>0.164375</td>
      <td>0.329171</td>
      <td>0.085221</td>
      <td>-0.014172</td>
      <td>-0.009091</td>
      <td>0.018297</td>
      <td>...</td>
      <td>-0.042192</td>
      <td>0.001860</td>
      <td>0.058311</td>
      <td>-0.013690</td>
      <td>-0.006113</td>
      <td>0.041178</td>
      <td>0.355061</td>
      <td>-0.265355</td>
      <td>0.120166</td>
      <td>0.301809</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>-0.254146</td>
      <td>0.092051</td>
      <td>0.066473</td>
      <td>-0.050027</td>
      <td>-0.672819</td>
      <td>0.077564</td>
      <td>0.332795</td>
      <td>-0.014351</td>
      <td>0.198804</td>
      <td>-0.113886</td>
      <td>...</td>
      <td>-0.012516</td>
      <td>0.008700</td>
      <td>-0.003088</td>
      <td>0.061881</td>
      <td>-0.013832</td>
      <td>-0.004364</td>
      <td>0.087350</td>
      <td>-0.023890</td>
      <td>-0.018085</td>
      <td>0.083422</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0.165476</td>
      <td>-0.192192</td>
      <td>-0.304780</td>
      <td>0.014116</td>
      <td>0.870678</td>
      <td>-0.243104</td>
      <td>-0.549199</td>
      <td>-0.065538</td>
      <td>-0.080224</td>
      <td>0.108924</td>
      <td>...</td>
      <td>-0.030261</td>
      <td>-0.032953</td>
      <td>-0.026403</td>
      <td>-0.072514</td>
      <td>0.023611</td>
      <td>0.131807</td>
      <td>-0.326487</td>
      <td>0.386262</td>
      <td>-0.300872</td>
      <td>-0.194207</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>0.198091</td>
      <td>0.139235</td>
      <td>0.213491</td>
      <td>0.033299</td>
      <td>-0.571176</td>
      <td>0.061643</td>
      <td>0.344935</td>
      <td>0.098379</td>
      <td>-0.100374</td>
      <td>-0.022950</td>
      <td>...</td>
      <td>0.080393</td>
      <td>0.045538</td>
      <td>0.013376</td>
      <td>0.042547</td>
      <td>-0.011742</td>
      <td>-0.162253</td>
      <td>0.157233</td>
      <td>-0.354649</td>
      <td>0.361247</td>
      <td>0.012893</td>
    </tr>
    <tr>
      <th>Officer</th>
      <td>0.162818</td>
      <td>0.028696</td>
      <td>-0.032631</td>
      <td>0.002231</td>
      <td>0.087288</td>
      <td>-0.013813</td>
      <td>-0.031316</td>
      <td>0.003678</td>
      <td>-0.003212</td>
      <td>-0.001202</td>
      <td>...</td>
      <td>0.006055</td>
      <td>-0.024048</td>
      <td>-0.017076</td>
      <td>-0.008281</td>
      <td>-0.003698</td>
      <td>-0.067030</td>
      <td>-0.026921</td>
      <td>0.013303</td>
      <td>0.003966</td>
      <td>-0.034572</td>
    </tr>
    <tr>
      <th>Royalty</th>
      <td>0.059466</td>
      <td>0.026214</td>
      <td>-0.030197</td>
      <td>0.004400</td>
      <td>-0.020408</td>
      <td>-0.010787</td>
      <td>0.033391</td>
      <td>0.077213</td>
      <td>-0.021853</td>
      <td>-0.054250</td>
      <td>...</td>
      <td>-0.012950</td>
      <td>-0.012202</td>
      <td>-0.008665</td>
      <td>-0.004202</td>
      <td>-0.001876</td>
      <td>-0.071672</td>
      <td>-0.023600</td>
      <td>0.008761</td>
      <td>-0.000073</td>
      <td>-0.017542</td>
    </tr>
    <tr>
      <th>Cabin_A</th>
      <td>0.125177</td>
      <td>0.020094</td>
      <td>-0.030707</td>
      <td>-0.002831</td>
      <td>0.047561</td>
      <td>-0.039808</td>
      <td>0.022287</td>
      <td>0.094914</td>
      <td>-0.042105</td>
      <td>-0.056984</td>
      <td>...</td>
      <td>-0.024952</td>
      <td>-0.023510</td>
      <td>-0.016695</td>
      <td>-0.008096</td>
      <td>-0.003615</td>
      <td>-0.242399</td>
      <td>-0.042967</td>
      <td>0.045227</td>
      <td>-0.029546</td>
      <td>-0.033799</td>
    </tr>
    <tr>
      <th>Cabin_B</th>
      <td>0.113458</td>
      <td>0.393743</td>
      <td>0.073051</td>
      <td>0.015895</td>
      <td>-0.094453</td>
      <td>-0.011569</td>
      <td>0.175095</td>
      <td>0.161595</td>
      <td>-0.073613</td>
      <td>-0.095790</td>
      <td>...</td>
      <td>-0.043624</td>
      <td>-0.041103</td>
      <td>-0.029188</td>
      <td>-0.014154</td>
      <td>-0.006320</td>
      <td>-0.423794</td>
      <td>0.032318</td>
      <td>-0.087912</td>
      <td>0.084268</td>
      <td>0.013470</td>
    </tr>
    <tr>
      <th>Cabin_C</th>
      <td>0.167993</td>
      <td>0.401370</td>
      <td>0.009601</td>
      <td>0.006092</td>
      <td>-0.077473</td>
      <td>0.048616</td>
      <td>0.114652</td>
      <td>0.158043</td>
      <td>-0.059151</td>
      <td>-0.101861</td>
      <td>...</td>
      <td>-0.053083</td>
      <td>-0.050016</td>
      <td>-0.035516</td>
      <td>-0.017224</td>
      <td>-0.007691</td>
      <td>-0.515684</td>
      <td>0.037226</td>
      <td>-0.137498</td>
      <td>0.141925</td>
      <td>0.001362</td>
    </tr>
    <tr>
      <th>Cabin_D</th>
      <td>0.132886</td>
      <td>0.072737</td>
      <td>-0.027385</td>
      <td>0.000549</td>
      <td>-0.057396</td>
      <td>-0.015727</td>
      <td>0.150716</td>
      <td>0.107782</td>
      <td>-0.061459</td>
      <td>-0.056023</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.034317</td>
      <td>-0.024369</td>
      <td>-0.011817</td>
      <td>-0.005277</td>
      <td>-0.353822</td>
      <td>-0.025313</td>
      <td>-0.074310</td>
      <td>0.102432</td>
      <td>-0.049336</td>
    </tr>
    <tr>
      <th>Cabin_E</th>
      <td>0.106600</td>
      <td>0.073949</td>
      <td>0.001084</td>
      <td>-0.008136</td>
      <td>-0.040340</td>
      <td>-0.027180</td>
      <td>0.145321</td>
      <td>0.027566</td>
      <td>-0.042877</td>
      <td>0.002960</td>
      <td>...</td>
      <td>-0.034317</td>
      <td>1.000000</td>
      <td>-0.022961</td>
      <td>-0.011135</td>
      <td>-0.004972</td>
      <td>-0.333381</td>
      <td>-0.017285</td>
      <td>-0.042535</td>
      <td>0.068007</td>
      <td>-0.046485</td>
    </tr>
    <tr>
      <th>Cabin_F</th>
      <td>-0.072644</td>
      <td>-0.037567</td>
      <td>0.020481</td>
      <td>0.000306</td>
      <td>-0.006655</td>
      <td>-0.008619</td>
      <td>0.057935</td>
      <td>-0.020010</td>
      <td>-0.020282</td>
      <td>0.030575</td>
      <td>...</td>
      <td>-0.024369</td>
      <td>-0.022961</td>
      <td>1.000000</td>
      <td>-0.007907</td>
      <td>-0.003531</td>
      <td>-0.236733</td>
      <td>0.005525</td>
      <td>0.004055</td>
      <td>0.012756</td>
      <td>-0.033009</td>
    </tr>
    <tr>
      <th>Cabin_G</th>
      <td>-0.085977</td>
      <td>-0.022857</td>
      <td>0.058325</td>
      <td>-0.045949</td>
      <td>-0.083285</td>
      <td>0.006015</td>
      <td>0.016040</td>
      <td>-0.031566</td>
      <td>-0.019941</td>
      <td>0.040560</td>
      <td>...</td>
      <td>-0.011817</td>
      <td>-0.011135</td>
      <td>-0.007907</td>
      <td>1.000000</td>
      <td>-0.001712</td>
      <td>-0.114803</td>
      <td>0.035835</td>
      <td>-0.076397</td>
      <td>0.087471</td>
      <td>-0.016008</td>
    </tr>
    <tr>
      <th>Cabin_T</th>
      <td>0.032461</td>
      <td>0.001179</td>
      <td>-0.012304</td>
      <td>-0.023049</td>
      <td>0.020558</td>
      <td>-0.013247</td>
      <td>-0.026456</td>
      <td>-0.014095</td>
      <td>-0.008904</td>
      <td>0.018111</td>
      <td>...</td>
      <td>-0.005277</td>
      <td>-0.004972</td>
      <td>-0.003531</td>
      <td>-0.001712</td>
      <td>1.000000</td>
      <td>-0.051263</td>
      <td>-0.015438</td>
      <td>0.022411</td>
      <td>-0.019574</td>
      <td>-0.007148</td>
    </tr>
    <tr>
      <th>Cabin_U</th>
      <td>-0.271918</td>
      <td>-0.507197</td>
      <td>-0.036806</td>
      <td>0.000208</td>
      <td>0.137396</td>
      <td>0.009064</td>
      <td>-0.316912</td>
      <td>-0.258257</td>
      <td>0.142369</td>
      <td>0.137351</td>
      <td>...</td>
      <td>-0.353822</td>
      <td>-0.333381</td>
      <td>-0.236733</td>
      <td>-0.114803</td>
      <td>-0.051263</td>
      <td>1.000000</td>
      <td>-0.014155</td>
      <td>0.175812</td>
      <td>-0.211367</td>
      <td>0.056438</td>
    </tr>
    <tr>
      <th>FamilySize</th>
      <td>-0.196996</td>
      <td>0.226465</td>
      <td>0.792296</td>
      <td>-0.031437</td>
      <td>-0.188583</td>
      <td>0.861952</td>
      <td>0.016639</td>
      <td>-0.036553</td>
      <td>-0.087190</td>
      <td>0.087771</td>
      <td>...</td>
      <td>-0.025313</td>
      <td>-0.017285</td>
      <td>0.005525</td>
      <td>0.035835</td>
      <td>-0.015438</td>
      <td>-0.014155</td>
      <td>1.000000</td>
      <td>-0.688864</td>
      <td>0.302640</td>
      <td>0.801623</td>
    </tr>
    <tr>
      <th>Family_Single</th>
      <td>0.116675</td>
      <td>-0.274826</td>
      <td>-0.549022</td>
      <td>0.028546</td>
      <td>0.284537</td>
      <td>-0.591077</td>
      <td>-0.203367</td>
      <td>-0.107874</td>
      <td>0.127214</td>
      <td>0.014246</td>
      <td>...</td>
      <td>-0.074310</td>
      <td>-0.042535</td>
      <td>0.004055</td>
      <td>-0.076397</td>
      <td>0.022411</td>
      <td>0.175812</td>
      <td>-0.688864</td>
      <td>1.000000</td>
      <td>-0.873398</td>
      <td>-0.318944</td>
    </tr>
    <tr>
      <th>Family_Small</th>
      <td>-0.038189</td>
      <td>0.197281</td>
      <td>0.248532</td>
      <td>0.002975</td>
      <td>-0.255196</td>
      <td>0.253590</td>
      <td>0.279855</td>
      <td>0.159594</td>
      <td>-0.122491</td>
      <td>-0.062909</td>
      <td>...</td>
      <td>0.102432</td>
      <td>0.068007</td>
      <td>0.012756</td>
      <td>0.087471</td>
      <td>-0.019574</td>
      <td>-0.211367</td>
      <td>0.302640</td>
      <td>-0.873398</td>
      <td>1.000000</td>
      <td>-0.183007</td>
    </tr>
    <tr>
      <th>Family_Large</th>
      <td>-0.161210</td>
      <td>0.170853</td>
      <td>0.624627</td>
      <td>-0.063415</td>
      <td>-0.077748</td>
      <td>0.699681</td>
      <td>-0.125147</td>
      <td>-0.092825</td>
      <td>-0.018423</td>
      <td>0.093671</td>
      <td>...</td>
      <td>-0.049336</td>
      <td>-0.046485</td>
      <td>-0.033009</td>
      <td>-0.016008</td>
      <td>-0.007148</td>
      <td>0.056438</td>
      <td>0.801623</td>
      <td>-0.318944</td>
      <td>-0.183007</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>32 rows × 32 columns</p>
</div>




```python
#查看各特征与生存情况的相关系数
CorrDf['Survived'].sort_values(ascending=False)
```




    Survived         1.000000
    Mrs              0.344935
    Miss             0.332795
    Pclass_1         0.285904
    Family_Small     0.279855
    Fare             0.257307
    Cabin_B          0.175095
    Embarked_C       0.168240
    Cabin_D          0.150716
    Cabin_E          0.145321
    Cabin_C          0.114652
    Pclass_2         0.093349
    Master           0.085221
    Parch            0.081629
    Cabin_F          0.057935
    Royalty          0.033391
    Cabin_A          0.022287
    FamilySize       0.016639
    Cabin_G          0.016040
    Embarked_Q       0.003650
    PassengerId     -0.005007
    Cabin_T         -0.026456
    Officer         -0.031316
    SibSp           -0.035322
    Age             -0.070323
    Family_Large    -0.125147
    Embarked_S      -0.149683
    Family_Single   -0.203367
    Cabin_U         -0.316912
    Pclass_3        -0.322308
    Sex             -0.543351
    Mr              -0.549199
    Name: Survived, dtype: float64



根据各个特征与生成情况（Survived）的相关系数大小，我们选择了这几个特征作为模型的输入：

头衔（前面所在的数据集titleDf）、客舱等级（pclassDf）、家庭大小（familyDf）、船票价格（Fare）、船舱号（cabinDf）、登船港口（embarkedDf）、性别（Sex）


```python
#特征选择
full_X=pd.concat([titleDf,#t头衔
                 pclassDf,#客舱等级
                 familyDf,#家庭大小
                 full['Fare'],#船票价格
                 cabinDf,#船舱号
                 embarkedDf,#登船港口
                 full['Sex']#性别
                 ],axis=1)
full_X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>FamilySize</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



## 4.构建模型

用训练数据和模型得到机器学习模型，用测试数据评估模型

### 4.1建立训练数据集和测试数据集


```python
'''
1）坦尼克号测试数据集因为是我们最后要提交给Kaggle的，里面没有生存情况的值，所以不能用于评估模型。
将Kaggle泰坦尼克号项目给我们的测试数据，叫做预测数据集（记为pred,也就是预测英文单词predict的缩写）。
也就是使用机器学习模型来对其生存情况就那些预测。
2）使用Kaggle泰坦尼克号项目给的训练数据集，做为我们的原始数据集（记为source），
从这个原始数据集中拆分出训练数据集（记为train：用于模型训练）和测试数据集（记为test：用于模型评估）。

'''
#原始数据有891行
SourceRow=rowNum_train

'''
sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。
'''
#原始数据集：特征
source_X=full_X.loc[0:SourceRow-1,:]
#原始数据集：标签
source_y=full.loc[0:SourceRow-1,'Survived']
#预测数据集：特征
pred_X=full_X.loc[SourceRow:,:]
```


```python
print('原始数据有：',source_X.shape)
print('预测数据有：',pred_X.shape)
```

    原始数据有： (891, 27)
    预测数据有： (418, 27)
    


```python
'''
从原始数据集（source）中拆分出训练数据集（用于模型训练train），测试数据集（用于模型评估test）
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
'''
from sklearn.model_selection import train_test_split
#建立模型用的训练数据集和测试数据集
train_X,test_X,train_y,test_y=train_test_split(source_X,source_y,train_size=0.8)
```


```python
#出数原始数据和测试数据大小
print('原始数据特征：',source_X.shape,
     '训练数据特征：',train_X.shape,
     '训练数据标签：',train_y.shape)
print('测试数据特征：',source_X.shape,
     '测试数据特征：',test_X.shape,
     '测试数据标签：',test_y.shape)
```

    原始数据特征： (891, 27) 训练数据特征： (712, 27) 训练数据标签： (712,)
    测试数据特征： (891, 27) 测试数据特征： (179, 27) 测试数据标签： (179,)
    


```python
#原始数据查看
source_y.head()
```




    0    0.0
    1    1.0
    2    1.0
    3    1.0
    4    0.0
    Name: Survived, dtype: float64



### 4.2选择机器学习算法


```python
#第1步：导入算法
from sklearn.linear_model import LogisticRegression
#第2步：创建模型：逻辑回归（logisic regression）
model = LogisticRegression()
```


```python
#随机森林Random Forests Model
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=100)

#支持向量机Support Vector Machines
#from sklearn.svm import SVC, LinearSVC
#model = SVC()

#Gradient Boosting Classifier
#from sklearn.ensemble import GradientBoostingClassifier
#model = GradientBoostingClassifier()

#K-nearest neighbors
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors = 3)

# Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
```

### 4.3 训练模型


```python
#训练模型
model.fit(train_X,train_y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



## 5.评估模型


```python
# 分类问题，score得到的是模型的正确率
model.score(test_X , test_y )
```




    0.78212290502793291



## 6.得到预测结果上传到kaggle中


```python
#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y = model.predict(pred_X)
'''
生成的预测值是浮点数（0.0,1,0）
但是Kaggle要求提交的结果是整型（0,1）
所以要对数据类型进行转换
'''
pred_Y=pred_Y.astype(int)
#乘客id
passenger_id=full.loc[SourceRow:,'PassengerId']
#数据框:乘客id,预测生存情况的值
predDf = pd.DataFrame( 
    { 'PassengerId': passenger_id , 
     'Survived': pred_Y } )
predDf.shape
predDf.head()
#保存结果
predDf.to_csv( 'titanic_pred.csv' , index = False )
```
