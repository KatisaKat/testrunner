import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

df_deals = pd.read_csv("cleaned_data_v2/cleaned_deals.csv") 

# some other groupings identified during model training
def clean_cat(cat):
    if ('metaverse' in cat):
        return 'ar'
    if ('paas' in cat):
        return 'platform'
    if ('geospatial' in cat):
        return 'geotech'
    return cat

df_deals['primaryTag'] = df_deals['primaryTag'].apply(clean_cat)

# creates a new dataframe of categories (sectors) with associated total investments and number of deals per year
cat_trends = df_deals.groupby(['primaryTag', 'year']).agg(
    totalInv=('amount', 'sum'), 
    numDeals=('amount', 'count')
).reset_index()

# calculates growth rate for both number of deals and investment amount
cat_trends['invGrowth'] = cat_trends.groupby('primaryTag')['totalInv'].pct_change()
cat_trends['dealGrowth'] = cat_trends.groupby('primaryTag')['numDeals'].pct_change()

cat_trends.replace([np.inf, -np.inf], np.nan, inplace=True)
cat_trends = cat_trends.dropna(subset=['invGrowth', 'dealGrowth']) # drop the first years where theres no growth rate indicators

# checking the quantiles for a good eval point
print(cat_trends['invGrowth'].quantile(0.5))
print(cat_trends['invGrowth'].quantile(0.6))
print(cat_trends['invGrowth'].quantile(0.75))

# target of high growth sector is a sector with more than 45% investment growth (top 0.6 quantile) and no decrease of deals from the prev year
cat_trends['hiGrowth'] = ((cat_trends['invGrowth'] > cat_trends['invGrowth'].quantile(0.6)) & (cat_trends['dealGrowth'] >= 0)).astype(int)
print(cat_trends)
cat_trends.to_csv("training_data.csv", index=False)  
cat_trends['hiGrowth'].value_counts().plot(kind='bar', color=['#3366FF', 'orange'], title='Class Distribution')
plt.show()

X = cat_trends.drop(columns=['hiGrowth', 'primaryTag'])
y = cat_trends['hiGrowth']

# text split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# test different models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(gamma="auto")
}
results = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1': {}
}

# train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results['accuracy'][name] = accuracy_score(y_test, y_pred)
    results['precision'][name] = precision_score(y_test, y_pred)
    results['recall'][name] = recall_score(y_test, y_pred)
    results['f1'][name] = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print (accuracy_score(y_test, y_pred))

best_model = max(results['accuracy'], key=results['accuracy'].get)
print(f"Best model: {best_model} with accuracy {results['accuracy'][best_model]:.4f}")

# plot metrics
labels = list(models.keys())
accuracy = [results['accuracy'][label] for label in labels]
precision = [results['precision'][label] for label in labels]
recall = [results['recall'][label] for label in labels]
f1 = [results['f1'][label] for label in labels]
x = np.arange(len(models))
width = 0.2 
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#9900CC')
ax.bar(x - 0.5*width, precision, width, label='Precision', color='#3366FF')
ax.bar(x + 0.5*width, recall, width, label='Recall', color='orange')
ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#FF5050')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.show()

# model - random forest classification
mod = RandomForestClassifier(n_estimators=100, random_state=42)
mod.fit(X_train, y_train)

# overfitting considerations: 
y_train_pred = mod.predict(X_train)
y_test_pred = mod.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')
# performing k fold cross validarion:
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
cv_scores = cross_val_score(mod, X, y, cv=cv, scoring='accuracy')
print(f"Cross-Validation Scores for each fold: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of Accuracy: {cv_scores.std():.4f}")


pred = []
# uses model to predict whether each sector (category) will be high growth or low growth in 2025
for sector in cat_trends['primaryTag'].unique():
    last_row = cat_trends[cat_trends['primaryTag'] == sector].iloc[-1]
    prev_year = last_row['year'] 
    prev_inv = last_row['totalInv']
    prev_deals = last_row['numDeals']
    prev_inv_growth = last_row['invGrowth']
    prev_deals_growth = last_row['dealGrowth']
    X_future = pd.DataFrame([[prev_year, prev_inv, prev_deals, prev_inv_growth, prev_deals_growth]],
                            columns=['year', 'totalInv', 'numDeals', 'invGrowth', 'dealGrowth'])
    X_future = X_future.to_numpy()
    hi_or_lo = mod.predict(X_future)[0]
    pred.append((sector, 2025, hi_or_lo))

future_df = pd.DataFrame(pred, columns=['primaryTag', 'year', 'hiGrowth'])
# sectors that are likely to be high growth in 2025 according to model
hg_sectors = future_df[future_df['hiGrowth'] == 1]
print(hg_sectors)

# model - random forest regression, investment amount prediction
X = cat_trends.drop(columns=['totalInv', 'primaryTag', 'hiGrowth'])
y = cat_trends['totalInv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mod = RandomForestRegressor(n_estimators=100, random_state=42)
mod.fit(X_train, y_train)
y_pred = mod.predict(X_test)

# model evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')
rangec = cat_trends['totalInv'].max() - cat_trends['totalInv'].min()
print(f'Range: {rangec}')
r2 = r2_score(y_test, y_pred)
print(f'R squared: {r2}')

# predicts the investment amount of each sector for 2025
predictions = []
for sector in cat_trends['primaryTag'].unique():
    last = cat_trends[cat_trends['primaryTag'] == sector].iloc[-1] 
    prev_year = last['year']
    prev_deals = last['numDeals']
    prev_dg = last['dealGrowth']
    prev_ig = last['invGrowth']
    X_future = pd.DataFrame([[prev_year, prev_deals, prev_ig, prev_dg]], 
                            columns=['year', 'numDeals', 'invGrowth', 'dealGrowth'])
    future_pred = mod.predict(X_future)[0]
    predictions.append((sector, 2025, future_pred))
    prev_inv = future_pred

future_df = pd.DataFrame(predictions, columns=['primaryTag', 'year', 'predInv'])
# top 5 predicted sectors with the most investments in 2025
print(future_df.sort_values(by='predInv', ascending=False).head(5))
