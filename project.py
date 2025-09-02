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

#vadimo top 8 rezisera, glumaca...
top_directors = df["director"].value_counts().nlargest(8).index
df["director"] = df["director"].where(df["director"].isin(top_directors), other="Other") 

top_actors = df["star"].value_counts().nlargest(8).index
df["star"] = df["star"].where(df["star"].isin(top_actors), other="Other") 

top_companies = df["company"].value_counts().nlargest(8).index
df["company"] = df["company"].where(df["company"].isin(top_companies), other="Other")

top_writers = df["writer"].value_counts().nlargest(8).index
df["writer"] = df["writer"].where(df["writer"].isin(top_writers), other="Other")

#dummies kolone za top 8
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

'''corr_matrix = X_train.corr()
# Pronađi parove kolona sa skoro 1 ili -1 korelacijom
threshold = 0.9999
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if np.isclose(corr_matrix.iloc[i, j], 1.0) or np.isclose(corr_matrix.iloc[i, j], -1.0):
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))
print("Highly correlated columns (possible perfect collinearity):")
for col1, col2, val in high_corr:
    print(f"{col1} <-> {col2} = {val}")'''

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

print(sm_model.summary())  # statistika, p-values, R^2  
print()
print()
print("Linearna regresija:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

def adjusted_r2(model, X, y):  
    """Compute adjusted R² for sklearn models"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2


# --- Priprema modela ---
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.01)

# --- Fitovanje Ridge ---
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

print()
print()
print("Ridge Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("Adjusted R2:", adjusted_r2(ridge_model, X_train, y_train))
print("\n")

# --- Fitovanje Lasso ---
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

print()
print()
print("Lasso Regression Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("Adjusted R2:", adjusted_r2(lasso_model, X_train, y_train))

'''OLS (LinearRegression + sm.OLS)

R² ≈ 0.685, Adjusted R² ≈ 0.681

Model objašnjava oko 68–69% varijanse zarade filma (nakon log-transformacije ciljne promenljive).

Ovo je solidno za ekonomske podatke jer zarada filmova može biti ekstremno različita (od malih nezavisnih filmova do blockbuster-a).

MAE ≈ 0.76, RMSE ≈ 1.06

MAE pokazuje prosečnu grešku u predikciji log(gross) od 0.76 jedinica.

RMSE je veći od MAE jer je osetljiviji na outliere (blokbasteri sa milijardama dolara).

Log-transformacija je dobra jer smanjuje uticaj ekstremnih vrednosti i stabilizuje model.

Pretpostavke linearne regresije:

Linearnost → zadovoljena (postoji linearna veza između feature-a i log(gross))

Autokorelacija reziduala → nema problema (Durbin-Watson ≈ 1.966)

Homoskedastičnost (jednaka varijansa reziduala) → zadovoljena (p≈0.38)

Normalnost reziduala → nije zadovoljena (p=0) → reziduali nisu normalno raspodeljeni

Ovo je očekivano zbog ekstremnih vrednosti u zaradi filmova.

Perfect collinearity → False → sada više nema savršene multikolinearnosti (odstranjena kolona “writer_Woody Allen”).

 Zaključak OLS:

Model predviđa relativno dobro (R² ~ 68%).

Koeficijenti su razumni, ali zbog nenormalnih reziduala i dummy varijabli, interpretacija može biti neprecizna.

Ako je cilj predikcija, OLS je dobar baseline, ali za stabilniju interpretaciju može se koristiti regularizacija.

Ridge Regression (L2 regularizacija)

MAE ≈ 0.76, RMSE ≈ 1.06, Adjusted R² ≈ 0.681

Rezultati su praktično identični OLS modelu.

Zašto Ridge?

Ridge ublažava multikolinearnost i smanjuje varijansu koeficijenata.

Kod ovog skupa podataka efekat nije dramatičan jer si već uklonila savršenu kolinearnost i većina dummy varijabli ima raznovrsne vrednosti.

 Zaključak Ridge:

Model je stabilniji od OLS u slučaju da bi bilo više kolinearnosti.

Predikcijske metrike su gotovo iste, što znači da regularizacija nije drastično poboljšala grešku ovde.

 Lasso Regression (L1 regularizacija)

MAE ≈ 0.77, RMSE ≈ 1.07, Adjusted R² ≈ 0.667

Metrike su blago lošije od Ridge i OLS.

Zašto Lasso?

Lasso može potpuno "ugasiti" (postaviti na 0) neke koeficijente i time obaviti feature selection.

Na dataset-u sa mnogo dummy varijabli, Lasso može ukloniti varijable koje ne doprinose mnogo predikciji, što ponekad povećava grešku.

Zaključak Lasso:

Pogodna za redukciju broja varijabli i stabilniju interpretaciju.

Blago gubi na predikcijskoj tačnosti u odnosu na OLS/Ridge, ali daje jednostavniji model.


OLS i Ridge daju vrlo slične rezultate, što znači da regularizacija nije bila nužna za stabilnost modela.

Lasso uklanja neke feature-e, ali blago gubi na tačnosti. Može biti koristan za smanjenje broja varijabli.

Pretpostavke linearne regresije su uglavnom zadovoljene osim normalnosti reziduala, što je očekivano zbog ekstrema u zaradi filmova.

Predikcija u log-skali je stabilna; vraćanjem u originalnu skalu (expm1(pred)) greška bi bila u dolarima i veća zbog heteroskedastičnosti.

 Praktični zaključak:

Za predikciju zarade filma, OLS ili Ridge su solidni.

Ako želiš pojednostavljen model sa manje feature-a, koristi Lasso.

Uvek imaj na umu da ekstremni blockbusteri mogu značajno uticati na grešku i normalnost reziduala.
'''






