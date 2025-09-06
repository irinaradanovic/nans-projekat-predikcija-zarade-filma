import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from utils_nans1 import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


def adjusted_r2(model, X, y):  
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

def plot_real_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predikcije")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color="red", linestyle="--", label="Idealna linija")
    plt.xlabel("Stvarne vrednosti (y_test)")
    plt.ylabel("Predviđene vrednosti (y_pred)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

df = pd.read_csv("movies.csv")
#izbacujemo year, release, country-nepotrebno
col_names =['rating','genre','score','votes','director','writer','star','budget','gross','company','runtime']
df = df[col_names]


#PRETPROCESIRANJE

#uklanjamo redove gde je budzet = 0
df = df[(df["budget"] > 0) & (df["gross"] > 0)]
df.dropna(subset=["budget", "gross"], inplace=True)

#print(check_for_missing_values(df))

# Brišemo redove gde nedostaje company ili rating - dataset je velik pa nece puno uticati
df = df.dropna(subset=["company", "rating"])
#nedostajuce vrednosti za runtime, popunjavamo medijanom
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

#dummies kolone za rating i genre, npr genre_Animation moze biti 0 ili 1
df = pd.get_dummies(df, columns=["rating", "genre"], drop_first=True, dtype=float)

#vadimo top 10 rezisera, glumaca...
top_directors = df["director"].value_counts().nlargest(10).index
df["director"] = df["director"].where(df["director"].isin(top_directors), other="Other") 

top_actors = df["star"].value_counts().nlargest(10).index
df["star"] = df["star"].where(df["star"].isin(top_actors), other="Other") 

top_companies = df["company"].value_counts().nlargest(10).index
df["company"] = df["company"].where(df["company"].isin(top_companies), other="Other")

top_writers = df["writer"].value_counts().nlargest(10).index
df["writer"] = df["writer"].where(df["writer"].isin(top_writers), other="Other")

#dummies kolone za top 10
df = pd.get_dummies(df, columns=["director", "writer", "star", "company"], drop_first=True, dtype=float)

'''pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)
print(df.head())   ''' #ispis kolona


X = df.drop("gross", axis=1)
y = np.log1p(df["gross"])  # log-transformacija ciljne promenljive (OVO JE URADJENO DA BI PRETPOSTAVKE (E, I) BILE ZADOVOLJENE )

# Log-transformacija numerickih kolona sa velikim rasponom
X[['budget','votes']] = np.log1p(X[['budget','votes']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#skaliranje numerickih kolona
num_cols = ["score", "votes", "budget", "runtime"] 
scaler = StandardScaler() 
X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) 
X_test[num_cols] = scaler.transform(X_test[num_cols])


#dobili smo da tamo gde je writer woody, on ce biti i director, izbacujemo u tom slucaju writer kolonu da ne bi imali savrsenu kolinearnost

X_train = X_train.drop(columns=["writer_Woody Allen"])
X_test = X_test.drop(columns=["writer_Woody Allen"])


#-----------MODEL LINEARNE REGRESIJE-----------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 


# Dodaj konstantu
X_train_const = sm.add_constant(X_train)
sm_model = sm.OLS(y_train, X_train_const).fit() 

print(sm_model.summary())  # statistika 
print()
print()
print("Linearna regresija test:")
print("Adjusted R2:", sm_model.rsquared_adj)




# Dodaj konstantu u X_train 
X_train_const = sm.add_constant(X_train, has_constant='add')

print()
print()
print("Pretpostavke:")
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



#------------RIDGE I LASSO------------

# --- Priprema modela ---
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.01)

# --- Fitovanje Ridge ---
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

print()
print()
print("Ridge Regression Results test:")
print("Adjusted R2:", adjusted_r2(ridge_model, X_test, y_test))
print("\n")

# --- Fitovanje Lasso ---
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

print()
print()
print("Lasso Regression Results test:")
print("Adjusted R2:", adjusted_r2(lasso_model, X_test, y_test))
print()


#------------GRADIENT BOOSTING------------

# Inicijalizacija modela
gb_model = GradientBoostingRegressor(
    n_estimators=300,      # broj stabala 
    learning_rate=0.05,    
    max_depth=4,           # dubina stabala
    random_state=42
)

# Treniranje
gb_model.fit(X_train, y_train)

# Predviđanje
y_pred_gb = gb_model.predict(X_test)

# Evaluacija
print()
print("Gradient Boosting Results test:")
print("Adjusted R2:", adjusted_r2(gb_model, X_test, y_test))
print()

# Važnost feature-a (feature importance)
importances = gb_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 8 najvažnijih feature-a (Gradient Boosting):")
for i in range(8):
    print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Grafički prikaz feature importance (Top 8)
plt.figure(figsize=(10,6))
plt.bar(range(8), importances[indices[:8]], align='center')
plt.xticks(range(8), X_train.columns[indices[:8]], rotation=45, ha="right")
plt.title("Top 8 Feature Importance - Gradient Boosting")
plt.tight_layout()
plt.show()



#------------RANDOM FOREST------------
rf_model = RandomForestRegressor(
    n_estimators=300,     # broj stabala 
    max_depth=10,      
    random_state=42,
    n_jobs=-1             
)

# Treniranje
rf_model.fit(X_train, y_train)

# Predviđanje
y_pred_rf = rf_model.predict(X_test)

# Evaluacija
print()
print("Random Forest Results test:")
print("Adjusted R2:", adjusted_r2(rf_model, X_test, y_test))
print()

# Važnost feature-a (feature importance)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 8 najvažnijih feature-a (Random Forest):")
for i in range(8):
    print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Grafički prikaz feature importance (Top 8)
plt.figure(figsize=(10,6))
plt.bar(range(8), importances[indices[:8]], align='center')
plt.xticks(range(8), X_train.columns[indices[:8]], rotation=45, ha="right")
plt.title("Top 8 Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

plt.close('all')

#------------DECISION TREE------------
dt_model = DecisionTreeRegressor(
    max_depth=10,       # možeš menjati dubinu stabla (npr. 5, 10, None)
    random_state=42
)

# Treniranje
dt_model.fit(X_train, y_train)

# Predviđanje
y_pred_dt = dt_model.predict(X_test)

# Evaluacija
print()
print("Decision Tree Results test:")
print("Adjusted R2:", adjusted_r2(dt_model, X_test, y_test))
print()

# Važnost feature-a (feature importance)
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 8 najvažnijih feature-a (Decision Tree):")
for i in range(8):
    print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Grafički prikaz feature importance (Top 8)
plt.figure(figsize=(10,6))
plt.bar(range(8), importances[indices[:8]], align='center')
plt.xticks(range(8), X_train.columns[indices[:8]], rotation=45, ha="right")
plt.title("Top 8 Feature Importance - Decision Tree")
plt.tight_layout()
plt.show()


#-------GRAFICKI PRIKAZ ZA SVAKI MODEL-------
# Linear Regression
plot_real_vs_predicted(y_test, y_pred, "Linear Regression: Real vs Predicted")

# Ridge
plot_real_vs_predicted(y_test, y_pred_ridge, "Ridge Regression: Real vs Predicted")

# Lasso
plot_real_vs_predicted(y_test, y_pred_lasso, "Lasso Regression: Real vs Predicted")

# Gradient Boosting
plot_real_vs_predicted(y_test, y_pred_gb, "Gradient Boosting: Real vs Predicted")

# Random Forest
plot_real_vs_predicted(y_test, y_pred_rf, "Random Forest: Real vs Predicted")

#Decision Tree
plot_real_vs_predicted(y_test, y_pred_dt, "Decision Tree: Real vs Predicted")


#Koliki je očekivani prihod filma sa budžetom 50M$, 100k glasova, 120 min, PG-13 rejtingom i Action žanrom?
#uradjeno na modelu OLS
# Primer novog filma
novi_film = pd.DataFrame({
    'budget': [50_000_000],   
    'votes': [100_000],       
    'runtime': [120],         
    'rating': ['PG-13'],     
    'genre': ['Action'],     
})


novi_film_encoded = pd.get_dummies(novi_film)
novi_film_encoded = novi_film_encoded.reindex(columns=X.columns, fill_value=0)


predicted_gross = sm_model.predict(novi_film_encoded)
print()
print(f"Ocekivani prihod: {predicted_gross[0]:,.2f} $")
