import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Global variables:
val_split = 0.1
test_split = 0.1

#Importing the dataframe from github and saving in data/raw folder.
url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
df = pd.read_csv(url)
df.to_csv('../data/raw/AB_NYC_2019.csv')

#df = pd.read_csv('../data/raw/AB_NYC_2019.csv')


#####Exploration and data cleaning#####
df.shape
df.info()

#Check for duplicates using ID column
df.duplicated("id").sum()

#Check for missing values
df.isna().sum()

#Heatmap to visualise missing values
sns.heatmap(df.isnull())

#Delete unnecessary columns
df.drop(['name', 'host_name', 'latitude', 'longitude'], axis=1, inplace=True)

#Statistical characteristics of numerical variables
df.describe().T

#Statistical characteristics of categorical variables
cat_variables = df.describe(include='object').T
cat_variables

percentage_ng = round((cat_variables.loc['neighbourhood_group', 'freq']/len(df) *100), 2)
percentage_rt = round((cat_variables.loc['room_type', 'freq']/len(df) * 100), 2)
print(f"The neighbourhood group with more listings is {cat_variables.loc['neighbourhood_group', 'top']} with {percentage_ng}% of the listings.")
print(f"The most common room type is {cat_variables.loc['room_type', 'top']} representing {percentage_rt}% of the listings.")

#####Analysis of univariate variables#####

###Categorical variables

#Plot of neighbourhood_group and room_type
fig, axis = plt.subplots(1, 2, figsize=(10,4))

sns.countplot(ax=axis[0], data=df, x="neighbourhood_group", color='#4B5320').set_xlabel('Neighbourhood Group')
sns.countplot(ax=axis[1], data=df, x="room_type", color='#000080').set_xlabel('Room Type')
fig.suptitle('Plot of Categorical Variables Neighbourhood Group and Room Type')
plt.tight_layout()
plt.show()

print(f"The dataset contains {df.neighbourhood.nunique()} unique neighbourhoods.")
print(f"The median of the neighbourhood column count is {df.neighbourhood.value_counts().median()}.")

#Plot of neighbourhoods
filtered_neighbourhoods = df['neighbourhood'].value_counts()[df['neighbourhood'].value_counts() > df.neighbourhood.value_counts().median()]
plt.figure(figsize=(17,6))
filtered_neighbourhoods.plot(kind='bar')
plt.xticks(fontsize=9)
plt.title('Histogram of the Neighbourhoods with a Count above the Median')
plt.tight_layout()
plt.show()

#Casting type of last review to datetime to extract month-year
df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')
df['lr_month_year'] = df['last_review'].dt.strftime('%Y-%m')

#Plot of last review date
df.sort_values(by='last_review', ascending=False, inplace=True)
fig=plt.figure(figsize=(14,6))
sns.countplot(data=df, x='lr_month_year')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()

###Numerical variables

num_variables = df.select_dtypes(include=np.number)
print(f"The numerical variables are: {num_variables.columns.values}.")

#Minimum nights
min_nights = df['minimum_nights'].value_counts()[df['minimum_nights'].value_counts() > df['minimum_nights'].value_counts().median()]

#Plots of minimum_nights
fig = plt.figure(figsize=(17,4))
min_nights.plot(kind='bar')
plt.title('Number of minimum nights required by hosts')
plt.tight_layout()
plt.xticks(fontsize=10)
plt.show()

#Price
df.price.describe()

#Plot of price
fig, axis=plt.subplots(1, 2, figsize=(8,3))
sns.histplot(ax=axis[0], data=df, x='price', bins=30, linewidth=0.25, edgecolor='white').set_xlim(0,2000)
sns.boxplot(ax=axis[1], data=df, x='price')
plt.tight_layout()
plt.show()

#Distplot of price with KDE
fig=plt.figure(figsize=(6,4))
sns.displot(data=df, x='price', kde=True)
plt.tight_layout()
plt.show()

#Avaialability_365
fig, axis = plt.subplots(2, 1, figsize=(7, 7))

sns.histplot(ax=axis[0], data=df, x='availability_365', linewidth=0.25, edgecolor='white')
sns.boxplot(ax=axis[1], data=df, x='availability_365')
fig.suptitle('Distribution of the avaialability_365 variable')
plt.tight_layout()
plt.show()

#Number of reviews, reviews per month and calculated host listings count
df[['number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']].describe().T

fig, axis=plt.subplots(3,2, figsize=(8,7))
sns.histplot(ax=axis[0,0], data=df, x='number_of_reviews', linewidth=0.25, edgecolor='white')
sns.boxplot(ax=axis[0,1], data=df, x='number_of_reviews')
sns.histplot(ax=axis[1,0], data=df, x='reviews_per_month', linewidth=0.25, edgecolor='white')
sns.boxplot(ax=axis[1,1], data=df, x='reviews_per_month')
sns.histplot(ax=axis[2,0], data=df, x='calculated_host_listings_count', linewidth=0.25, edgecolor='white')
sns.boxplot(ax=axis[2,1], data=df, x='calculated_host_listings_count')
plt.tight_layout()
plt.show()

#####Analysis of multivariate variables#####

##Price, Room Type and Neighbourhood Group

fig, axis = plt.subplots(2,2, figsize=(10,7))
sns.barplot(ax=axis[0,0], x=df.room_type, y=df.price, palette='Set2').set_xlabel('Room Type')
sns.barplot(ax=axis[0,1], x=df.neighbourhood_group, y=df.price, palette='Set3').set_xlabel('Neighbourhood Group')
sns.barplot(ax=axis[1,0],data=df, x="neighbourhood_group", y="price", hue="room_type", palette="inferno")
sns.countplot(ax=axis[1,1], x='neighbourhood_group', hue="room_type", data = df, palette = "CMRmap")
axis[1,0].legend(loc='best', fontsize=8)
axis[1,0].tick_params(axis='x', labelsize=9)
axis[0,1].tick_params(axis='x', labelsize=9)
axis[1,1].legend(loc='best', fontsize=8)
axis[1,1].tick_params(axis='x', labelsize=9)
plt.tight_layout()
plt.show()

#Price x minimum nights, price x number of reviews, price x reviews per month, price x calculated host listings count
fig, axis = plt.subplots(1,4, figsize=(12,4))
sns.scatterplot(ax=axis[0], data=df, x='price', y='minimum_nights')
sns.scatterplot(ax=axis[1], data=df, x='price', y='number_of_reviews')
sns.scatterplot(ax=axis[2], data=df, x='price', y='reviews_per_month')
sns.scatterplot(ax=axis[3], data=df, x='price', y='calculated_host_listings_count')
plt.tight_layout()
plt.show()

#Correlation analysis

#Factorizing neighbourhood_group and room_type
df['neighbourhood_group_N'] = pd.factorize(df['neighbourhood_group'])[0]
df['room_type_N'] = pd.factorize(df['room_type'])[0]

#Correlation matrix
corr_matrix = df[['neighbourhood_group_N', 'room_type_N',
       'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365']].corr()
sns.heatmap(corr_matrix, annot=True,fmt = ".2f")
plt.show()

#Distplot of numerical variables
variables = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group_N', 'room_type_N']
num_plots = len(variables)
total_cols = 4
total_rows = len(variables)//total_cols
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(6*total_cols, 6*total_rows), constrained_layout=True)

index = 0
for col in variables:

    row = index //total_cols
    pos = index % total_cols
    print(row, pos)
    sns.distplot(df[col], kde=True, rug = False, ax=axs[row][pos])
    index += 1

plt.show()
plt.tight_layout()

#Regplot of neighbourhood_group per price and room_type per price
fig, axis = plt.subplots(figsize=(10,5), ncols=2)
sns.regplot(ax=axis[0], data = df, x = "price", y = "room_type_N")
sns.regplot(ax=axis[1], data = df, x = "price", y = "neighbourhood_group_N")
plt.tight_layout()
plt.show()

#####Feature scaling#####

#New dataframe
df_encoded = pd.DataFrame()
df_encoded['room_type'] = pd.factorize(df['room_type'])[0]
dummies = pd.get_dummies(df['neighbourhood_group'], prefix='neighbourhood_group')
df_encoded = pd.concat([df_encoded, dummies], axis=1)

#Normalisation
num_to_normalise = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
scaler = MinMaxScaler()
norm_features = scaler.fit_transform(df[num_to_normalise])
df_encoded = pd.concat([df_encoded, pd.DataFrame(norm_features, index = df.index, columns = num_to_normalise)], axis=1)

df_encoded.shape

#Splitting the dataframe for test and train
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_split, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = test_split, random_state = 42)

##Exporting data processed to csv
df_encoded.to_csv('../data/processed/AB_NYC_2019_processed.csv')
