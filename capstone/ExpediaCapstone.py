from time import time
t = time
print 'Import statements'
t0 = time()
import math as math
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.naive_bayes import BernoulliNB
from datetime import datetime
print round(time()-t0,3),"s"

print 'Training dtypes'
t0 = time()
train_dtypes = {'date_time': pd.np.object,
'site_name': pd.np.int64,
'posa_continent': pd.np.int64,
'user_location_country': pd.np.int64,
'user_location_region': pd.np.int64,
'user_location_city': pd.np.int64,
'orig_destination_distance': pd.np.float64,
'user_id': pd.np.int64,
'is_mobile': pd.np.int64,
'is_package': pd.np.int64,
'channel': pd.np.int64,
'srch_ci': pd.np.object,
'srch_co': pd.np.object,
'srch_adults_cnt': pd.np.int64,
'srch_children_cnt': pd.np.int64,
'srch_rm_cnt': pd.np.int64,
'srch_destination_id': pd.np.int64,
'srch_destination_type_id': pd.np.int64,
'is_booking': pd.np.int64,
'cnt': pd.np.int64,
'hotel_continent': pd.np.int64,
'hotel_country': pd.np.int64,
'hotel_market': pd.np.int64,
'hotel_cluster': pd.np.int64}
print round(time()-t0,3),"s"

print 'Importing training data'
t0 = time()
all_train = pd.read_csv('train.csv', dtype=train_dtypes)
#temp_train = pd.read_csv('train.csv', dtype=train_dtypes, iterator=True, chunksize=1000)
#all_train = pd.concat(temp_train, ignore_index=True)
print round(time()-t0,3),"s"

print 'Adjusting training dataframe columns'
t0 = time()
all_train['orig_destination_distance'] = all_train['orig_destination_distance'].fillna(all_train['orig_destination_distance'].median()).astype(int)
#all_train['date_time'] = pd.to_datetime(all_train['date_time'], errors='coerce')
#all_train['srch_ci'] = pd.to_datetime(all_train['srch_ci'], errors='coerce')
#all_train['srch_co'] = pd.to_datetime(all_train['srch_co'], errors='coerce')
#Remove dates columns
all_train = all_train.drop(['date_time','srch_ci','srch_co','is_booking', 'cnt'], 1)
print round(time()-t0,3),"s"

print 'Creating smaller training dataset'
t0 = time()
#split = int(0.75*len(all_train))
#all_train = all_train[split:]
#train = all_train[0:split]
#test  = all_train[split:]
print round(time()-t0,3),"s"

print 'Separating features from labels'
t0 = time()
features_train = all_train.ix[:,:'hotel_market'] 
labels_train = all_train.ix[:,'hotel_cluster':]
#features_test = test.ix[:,:'hotel_market'] 
#labels_test = test.ix[:,'hotel_cluster':]
features_train = features_train.values
labels_train = labels_train.values
#features_test = features_test.values
#labels_test = labels_test.values
print round(time()-t0,3),"s"


print 'Fitting classifier'
t0 = time()
#clf = GaussianNB()
#clf = ensemble.AdaBoostClassifier(n_estimators=10)
#clf = ensemble.AdaBoostClassifier(SVC(probability=True, kernel='linear'),n_estimators=10)
#clf = ensemble.GradientBoostingClassifier(SVC(probability=True, kernel='linear'),n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#clf = ensemble.RandomForestClassifier(n_estimators=10)
#clf - BernoulliNB()
clf = clf.fit(features_train, labels_train.ravel())
print round(time()-t0,3),"s"

#t0 = time()
#pred = clf.predict(features_test)


#print accuracy_score(pred, labels_test.ravel())

print 'Testing dtypes'
t0 = time()
test_dtypes = {'id': pd.np.int64,
'date_time': pd.np.object,
'site_name': pd.np.int64,
'posa_continent': pd.np.int64,
'user_location_country': pd.np.int64,
'user_location_region': pd.np.int64,
'user_location_city': pd.np.int64,
'orig_destination_distance': pd.np.float64,
'user_id': pd.np.int64,
'is_mobile': pd.np.int64,
'is_package': pd.np.int64,
'channel': pd.np.int64,
'srch_ci': pd.np.object,
'srch_co': pd.np.object,
'srch_adults_cnt': pd.np.int64,
'srch_children_cnt': pd.np.int64,
'srch_rm_cnt': pd.np.int64,
'srch_destination_id': pd.np.int64,
'srch_destination_type_id': pd.np.int64,
'hotel_continent': pd.np.int64,
'hotel_country': pd.np.int64,
'hotel_market': pd.np.int64}
print round(time()-t0,3),"s"

print 'Importing testing data'
t0 = time()
all_test = pd.read_csv('test.csv', dtype=test_dtypes)
print round(time()-t0,3),"s"

print 'Adjusting testing data columns'
t0 = time()
all_test['orig_destination_distance'] = all_test['orig_destination_distance'].fillna(0.0).astype(int)
#all_train['date_time'] = pd.to_datetime(all_train['date_time'], errors='coerce')
#all_train['srch_ci'] = pd.to_datetime(all_train['srch_ci'], errors='coerce')
#all_train['srch_co'] = pd.to_datetime(all_train['srch_co'], errors='coerce')
#Remove dates columns
testing_file = all_test.drop(['id', 'date_time','srch_ci','srch_co'], 1)
testing_file = testing_file.values
print round(time()-t0,3),"s"

print 'Predicting label probabilities'
t0 = time()
probs = pd.DataFrame(clf.predict_proba(testing_file))
print round(time()-t0,3),"s"

print 'Predicting label probabilities'
t0 = time()
probs_series = pd.Series([(i, r.sort_values(ascending=False)[:5].index.values) for i,r in probs.iterrows()])
probs_series = probs_series.values
print round(time()-t0,3),"s"

print 'Creating indices and values for export'
t0 = time()
indices = [a for a,b in probs_series]
values = [b for a,b in probs_series]
print round(time()-t0,3),"s"

#t0 = time()
#pred = clf.predict(testing_file)

print 'Creating submission dataframe'
t0 = time()
submission = pd.DataFrame()
submission['id'] = indices
submission['hotel_cluster'] = [' '.join(str(x) for x in y) for y in values]
print round(time()-t0,3),"s"

print 'Exporting submission dataframe'
t0 = time()
submission.to_csv('submission.csv', index=False)
print round(time()-t0,3),"s"

print round(math.ceil((time()-t)/60),2),'minutes'