import pandas as pd
import pickle
from sklearn import tree

df = pd.read_csv('KvsT.csv')
print(df.head(10))

print(df['身長'].head(10))
print(df['派閥'].head(10))
print(type(df['派閥']))

# 特徴量を変数 x に代入
xcol = ['身長','体重','年代']
x = df[xcol]
print(x)

# 正解データを変数 t に代入
t = df['派閥']
print(t)

# モデルの準備（未学習）
model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(x,t)
print("モデルの学習を実行しました")

sample1 = [[170,70,20]]
print(model.predict(sample1))

sample2 = [[178,70,34]]
print(model.predict(sample2))

# モデルの評価
print(model.score(x,t))

# モデルの保存
with open('KinokoTakenoko.pkl','wb') as f:
    pickle.dump(model,f)
