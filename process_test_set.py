import dask.dataframe as dd
import numpy as np

test_file = 'test/test_set.csv'

test = dd.read_csv(test_file)

zscore = lambda x: (x - x.mean()) / x.std()

test['dt'] = test.groupby(['object_id','passband'])['mjd'].apply(lambda x: x.diff()).fillna(-10)

test['x'] = np.log(1+np.power(test['flux']/test['flux_err'],2))
test['x'] = test.groupby(['object_id','passband'])['x'].apply(zscore)

cols_to_drop = ['mjd', 'flux', 'flux_err']
test = test.drop(cols_to_drop, 1)

test = test.groupby(['object_id', 'passband']).apply(
    lambda x: x.set_index(['object_id', 'passband']).to_dict(orient='list')
)

test = test.compute()
test = test.unstack('passband')

test.to_csv('test_.csv')