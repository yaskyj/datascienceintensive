from time import time
t = time()
print 'Import statements'
t0 = time()
import pandas as pd
import numpy as np
import gc
#from sklearn import cross_validation
from sklearn import ensemble
from sklearn.decomposition import RandomizedPCA
#from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import SelectKBest
#from sklearn.naive_bayes import GaussianNB
from sklearn import tree
#from sklearn.svm import SVC
#from sklearn.naive_bayes import BernoulliNB
#from datetime import datetime
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

print 'Training import'
t0 = time()
all_train = pd.read_csv('train.csv', dtype=train_dtypes, iterator=True, chunksize=1000)
#iter_csv = pandas.read_csv('file.csv', iterator=True, chunksize=1000)
#all_train = pd.concat([chunk[chunk['is_booking'] == 1] for chunk in all_train])
all_train = pd.concat(all_train, ignore_index=True)
#temp_train = pd.read_csv('train.csv', dtype=train_dtypes, iterator=True, chunksize=1000)
#all_train = pd.concat(temp_train, ignore_index=True)
print round(time()-t0,3),"s"

print 'Training features'
t0 = time()
all_train['id'] = [i for i in range(0, len(all_train))]
all_train['orig_destination_distance'] = all_train['orig_destination_distance'].fillna(-1)
all_train['date_time'] = pd.to_datetime(all_train['date_time'], errors='coerce')
all_train['srch_ci'] = pd.to_datetime(all_train['srch_ci'], errors='coerce')
all_train['srch_co'] = pd.to_datetime(all_train['srch_co'], errors='coerce')
all_train['activity_month'] = all_train['date_time'].fillna(-1).dt.month.astype(int)
all_train['checkin_month'] = all_train['srch_ci'].fillna(-1).dt.month.astype(int)
all_train['checkout_month'] = all_train['srch_co'].fillna(-1).dt.month.astype(int)
#Split groups into two different classifiers for destinations vs. no destinations
print round(time()-t0,3),"s"

print 'Destinations import'
t0 = time()
destinations = pd.read_csv('destinations.csv')
print round(time()-t0,3),"s"

print 'Destinations files creationg'
t0 = time()
destination_ids = destinations['srch_destination_id']
destination_ds = destinations.drop(['srch_destination_id'], 1)
print round(time()-t0,3),"s"

print 'Destination PCA fit'
t0 = time()
pca = RandomizedPCA(n_components=1, whiten=True).fit(destination_ds)
print round(time()-t0,3),"s"

print 'Destinations PCA transform'
t0 = time()
destinations_pca = pca.transform(destination_ds)
print round(time()-t0,3),"s"

print 'Destinations dataframe creation'
t0 = time()
destinations_df = pd.DataFrame()
destinations_df['srch_destination_id'] = destination_ids
destinations_df['latent_destinations'] = destinations_pca
print round(time()-t0,3),"s"

print 'Destinations merge with training'
t0 = time()
with_dest_match = pd.merge(all_train, destinations_df)
print round(time()-t0,3),"s"

print 'Training with no destinations match'
t0 = time()
wo_dest_match = all_train[~(all_train.id.isin(with_dest_match.id))]
print round(time()-t0,3),"s"

print 'Both training sets feature manipulation'
t0 = time()
with_features = with_dest_match.drop(['id', 'is_booking', 'cnt', 'user_id', 'hotel_cluster', 'date_time', 'srch_ci', 'srch_co'],1)
with_labels = with_dest_match['hotel_cluster']
wo_features = wo_dest_match.drop(['id', 'is_booking', 'cnt', 'user_id', 'hotel_cluster', 'date_time', 'srch_ci', 'srch_co'],1)
wo_labels = wo_dest_match['hotel_cluster']
with_features = with_features.reindex_axis(sorted(with_features.columns), axis=1)
wo_features = wo_features.reindex_axis(sorted(wo_features.columns), axis=1)
print round(time()-t0,3),"s"

print 'Delete all_train dataframe'
t0 = time()
del all_train
del destinations
gc.collect()
print round(time()-t0,3),"s"

print 'Fitting classification model'
t0 = time()
#clf = LogisticRegression(tol=0.1)
#clf = GaussianNB()
#clf_with = ensemble.AdaBoostClassifier().fit(with_features, with_labels.values.ravel())
#clf_wo = ensemble.AdaBoostClassifier().fit(wo_features, wo_labels.values.ravel())
#clf = ensemble.AdaBoostClassifier(SVC(probability=True, kernel='linear'),n_estimators=10)
#clf = ensemble.GradientBoostingClassifier(SVC(probability=True, kernel='linear'),n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, verbose=3)
# clf_with = tree.DecisionTreeClassifier(min_samples_split=100).fit(with_features, with_labels.values.ravel())
# clf_wo = tree.DecisionTreeClassifier(min_samples_split=50).fit(wo_features, wo_labels.values.ravel())
clf_with = ensemble.RandomForestClassifier(n_estimators=50, min_samples_split=500, n_jobs=-1, max_depth=10).fit(with_features, with_labels.values.ravel())
clf_wo = ensemble.RandomForestClassifier(n_estimators=50, min_samples_split=500, n_jobs=-1, max_depth=10).fit(wo_features, wo_labels.values.ravel())
#clf - BernoulliNB()
print round((time()-t0)/60,2),"minutes"

print 'Delete all training dataframes'
t0 = time()
del with_features
del with_labels
del wo_features
del wo_labels
gc.collect()
print round((time()-t0),3),"s"

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

print 'Import testing'
t0 = time()
all_test = pd.read_csv('test.csv', dtype=test_dtypes)
print round(time()-t0,3),"s"

print 'Testing feature manipulation'
t0 = time()
all_test['orig_destination_distance'] = all_test['orig_destination_distance'].fillna(-1)
all_test['date_time'] = pd.to_datetime(all_test['date_time'], errors='coerce')
all_test['srch_ci'] = pd.to_datetime(all_test['srch_ci'], errors='coerce')
all_test['srch_co'] = pd.to_datetime(all_test['srch_co'], errors='coerce')
all_test['activity_month'] = all_test['date_time'].fillna(-1).dt.month.astype(int)
all_test['checkin_month'] = all_test['srch_ci'].fillna(-1).dt.month.astype(int)
all_test['checkout_month'] = all_test['srch_co'].fillna(-1).dt.month.astype(int)
print round(time()-t0,3),"s"

print 'Testing file with destinations'
t0 = time()
with_dest_test = pd.merge(all_test, destinations_df)
with_testing_ids = with_dest_test['id'] 
with_testing_features = with_dest_test.drop(['user_id', 'id', 'date_time', 'srch_ci', 'srch_co'],1)
print round(time()-t0,3),"s"

print 'Testing file without destinations'
t0 = time()
wo_dest_test = all_test[~(all_test.id.isin(with_dest_test.id))]
wo_testing_ids = wo_dest_test['id']
wo_testing_features = wo_dest_test.drop(['user_id', 'id', 'date_time', 'srch_ci', 'srch_co'],1)
print round(time()-t0,3),"s"

print 'Reindex both testing files'
t0 = time()
with_testing_features = with_testing_features.reindex_axis(sorted(with_testing_features.columns), axis=1)
wo_testing_features = wo_testing_features.reindex_axis(sorted(wo_testing_features.columns), axis=1)
print round(time()-t0,3),"s"

print 'Delete all_test dataframe'
t0 = time()
del all_test
gc.collect()
print round((time()-t0),3),"s"

print 'Predict probablities'
t0 = time()
#feature_test_file = selector.transform(all_test)
#pred = clf.predict(feature_test_list)
with_test_probs = pd.DataFrame(clf_with.predict_proba(with_testing_features))
wo_test_probs = pd.DataFrame(clf_wo.predict_proba(wo_testing_features))
print round(time()-t0,3),"s"

print 'Manipulate probablities with destinations'
t0 = time()
with_test_probs = pd.Series([(i, r.sort_values(ascending=False)[:5].index.values) for i,r in with_test_probs.iterrows()])
with_test_probs = with_test_probs.values
print round(time()-t0,3),"s"

print 'Creating series for with destinations'
t0 = time()
indices_1 = with_testing_ids.values
values_1 = [b for a,b in with_test_probs]
print round(time()-t0,3),"s"

print 'Creating DataFrame for with destinations'
t0 = time()
submission_1 = pd.DataFrame()
submission_1['id'] = indices_1
submission_1['hotel_cluster'] = [' '.join(str(x) for x in y) for y in values_1]
print round(time()-t0,3),"s"

print 'Manipulate probablities without destinations'
t0 = time()
wo_test_probs = pd.Series([(i, r.sort_values(ascending=False)[:5].index.values) for i,r in wo_test_probs.iterrows()])
wo_test_probs = wo_test_probs.values
print round(time()-t0,3),"s"

print 'Creating series for without destinations'
t0 = time()
indices_2 = wo_testing_ids.values
values_2 = [b for a,b in wo_test_probs]
print round(time()-t0,3),"s"

print 'Creating dataframe for with destinations'
t0 = time()
submission_2 = pd.DataFrame()
submission_2['id'] = indices_2
submission_2['hotel_cluster'] = [' '.join(str(x) for x in y) for y in values_2]
print round(time()-t0,3),"s"

print 'Creating submission dataframe'
t0 = time()
submission = pd.concat([submission_1, submission_2])
print round(time()-t0,3),"s"

print 'Sorting submissions by id'
t0 = time()
submission.sort_values(by='id', inplace=True)
print round(time()-t0,3),"s"

print 'Exporting submission'
t0 = time()
submission.to_csv('submission.csv', index=False)
print round(time()-t0,3),"s"

print round(((time()-t)/60),2),"minutes"