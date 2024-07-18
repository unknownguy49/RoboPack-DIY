import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load DataFrame
df = pd.read_csv('/workspaces/RoboPack-DIY/House Price Prediction/house_prices.csv')

# Visualize Data Features with Bar Plot
df.hist(bins=50, figsize=(20,15))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Prepare Data
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and Evaluate
pred = lr.predict(X_test)
mse = mean_squared_error(y_test, pred)
print('Mean Squared Error:', mse)