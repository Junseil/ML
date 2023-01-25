import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# 데이터 링크
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# # 데이터 다운로드
# def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
#     os.makedirs(housing_path, exist_ok=True)
#     tgz_path = os.path.join(housing_path, "hosing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path = housing_path)
#     housing_tgz.close()

# 다운로드된 데이터프레임 반환
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# fetch_housing_data()
housing = load_housing_data()

# # 데이터프레임 확인
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
#
# # 히스토그램
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# # 테스트세트 분리
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# # 테스트세트 분리예시
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set))
# print(len(test_set))

# # 파이썬 2, 3 호환성을 위한 비트연산
# # 책에서는 test_set_check,,, 그대로 입력시 자동으로 pytest 적용되어 오류생김
# def tes_set_check(identifier, test_ratio):
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

# # test 셋을 고정하면서 분리
# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: tes_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

# # 식별자로 index를 사용하여 분리를 고정시켜서 분리해볼 수 있음
# housing_with_id = housing.reset_index()
# # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# # 행이 바뀌어도 유지되게 위도 경도로 식별자 생성
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# # 사이킷런에서 제공하는 샘플링방식
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#
# 계층적 샘플링
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# # 비율 확인
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# 데이터 원상복귀 - 추가했던 income_cat 행 제거
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()

# # 상관관계 조사
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)
#
# from pandas.plotting import scatter_matrix
#
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
#
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)
#
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedroom_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
#
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# # 결측치 해결방법 3가지 1. 결측치 구역제거 2. 전체 특성제거 3. 중간값 대입
# housing.dropna(subset=["total_bedrooms"])
# housing.drop("total_bedrooms", axis=1)
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

# 중간값을 statistics_ 에 저장
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

# imputer 한 넘파이 성격 데이터를 판다스 데이터 프레임으로 변경
housing_tr = pd.DataFrame(imputer.transform(housing_num), columns=housing_num.columns,
                          index=housing_num.index)

# 데이터 성격비교
# print(imputer.transform(housing_num))
# print(housing_tr)

# 텍스트 특성 확인
housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(10))

# 텍스트를 수치에 각각 대응시킴
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# housing_cat_encoded[:10]
# print(ordinal_encoder.categories_)


# 수치의 거리개념이 반영될 수 있으므로 one-hot 벡터로 변경
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)

# 나만의 변환기 만들기
rooms_ix, bedroom_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

