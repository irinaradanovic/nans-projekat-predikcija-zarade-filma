import pandas as pd
import ast
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from utils_nans1 import *
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

df = pd.read_csv("movies.csv")
col_names =['rating','genre','score','votes','director','writer','star','budget','gross','company','runtime']
df = df[col_names]
#uklanjamo redove gde je budzet = 0
df = df[(df["budget"] > 0) & (df["gross"] > 0)]
df.dropna(subset=["budget", "gross"], inplace=True)

#print(check_for_missing_values(df))



# Brišemo redove gde nedostaje company ili rating
df = df.dropna(subset=["company", "rating"])
#nedostajuce vrednosti za runtime, popunjavamo medijanom
df["runtime"] = df["runtime"].fillna(df["runtime"].median())


df = pd.get_dummies(df, columns=["rating", "genre"], drop_first=True, dtype=float)

top_directors = df["director"].value_counts().nlargest(10).index
df["director"] = df["director"].where(df["director"].isin(top_directors), other="Other") 

top_actors = df["star"].value_counts().nlargest(10).index
df["star"] = df["star"].where(df["star"].isin(top_actors), other="Other") 

top_companies = df["company"].value_counts().nlargest(10).index
df["company"] = df["company"].where(df["company"].isin(top_companies), other="Other")

top_writers = df["writer"].value_counts().nlargest(10).index
df["writer"] = df["writer"].where(df["writer"].isin(top_writers), other="Other")

df = pd.get_dummies(df, columns=["director", "writer", "star", "company"], drop_first=True, dtype=float)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)
print(df.head())


X = df.drop("gross", axis=1)
#y = df["gross"]  #ovo prediktujemo
y = np.log1p(df["gross"])  # log-transformacija ciljne promenljive

# Log-transformacija numeričkih kolona sa velikim rasponom
X[['budget','votes']] = np.log1p(X[['budget','votes']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
num_cols = ["score", "votes", "budget", "runtime"] 
scaler = StandardScaler() 
X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) 
X_test[num_cols] = scaler.transform(X_test[num_cols])

'''# Skaliranje numeričkih kolona
num_cols = ["score", "votes", "budget", "runtime"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 3. Podela na trening i test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) '''


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 


# Dodaj konstantu
X_train_const = sm.add_constant(X_train)
sm_model = sm.OLS(y_train, X_train_const).fit() 

print(sm_model.summary())  # statistika, p-values, R^2  

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Dodaj konstantu u X_train (da bude 68 kolona kao u modelu)
X_train_const = sm.add_constant(X_train, has_constant='add')


print("Assumptions satisfied?:", are_assumptions_satisfied(sm_model, X_train_const, y_train))
is_linearity_found, p_value = linear_assumption(sm_model, X_train_const, y_train)
print("Is linearity found: ", is_linearity_found, "p_value: ", p_value)
autocorr, dw = independence_of_errors_assumption(sm_model, X_train_const, y_train)
print("Autocorrelation:", autocorr, "Durbin-Watson:", dw)

dist_type, p_val = normality_of_errors_assumption(sm_model, X_train_const, y_train)
print("Residuals distribution:", dist_type, "p-value:", p_val)

eq_var, p_val_eq = equal_variance_assumption(sm_model, X_train_const, y_train)
print("Equal variance:", eq_var, "p-value:", p_val_eq)

has_collinearity = perfect_collinearity_assumption(X_train_const)
print("Perfect collinearity:", has_collinearity)


'''# 8. Vizualizacija stvarna vs predikcija
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Stvarna zarada")
plt.ylabel("Predikcija")
plt.title("OLS Regression: Stvarna vs Predikcija")
plt.show()

# 9. Residuali
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sn.histplot(residuals, kde=True)
plt.title("Distribucija residuala")
plt.show()   '''

'''plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarna zarada")
plt.ylabel("Predikcija")
plt.title("Linear Regression: stvarna vs predikcija")
plt.show()


residuals = y_test - y_pred
sn.histplot(residuals, kde=True)
plt.title("Distribucija residuala")
plt.show()    '''






