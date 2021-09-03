# -*- coding:utf-8 -*-
"""

"""
import time

from sklearn.model_selection import train_test_split

from deeptables.datasets import dsutils
from deeptables.models.hyper_dt import HyperDT, default_dt_space, mini_dt_space
from deeptables.utils import consts
from hypernets.core.callbacks import SummaryCallback, FileStorageLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.trial import DiskTrialStore
from hypernets.searchers.evolution_searcher import EvolutionSearcher

homedir = f'{consts.PROJECT_NAME}_run_dt_{time.strftime("%Y%m%d%H%M%S")}'
disk_trial_store = DiskTrialStore(f'{homedir}/trial_store')

# searcher = MCTSSearcher(mini_dt_space, max_node_space=0,optimize_direction=OptimizeDirection.Maximize)
# searcher = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize)
searcher = EvolutionSearcher(mini_dt_space, 200, 100, regularized=True, candidates_size=30,
                             optimize_direction=OptimizeDirection.Maximize)

hdt = HyperDT(searcher,
              callbacks=[SummaryCallback(), FileStorageLoggingCallback(searcher, output_dir=f'{homedir}/hyn_logs')],
              reward_metric='AUC',
              earlystopping_patience=1)

space = mini_dt_space()
assert space.combinations == 589824
space2 = default_dt_space()
assert space2.combinations == 3559292928

df = dsutils.load_adult()
# df.drop(['id'], axis=1, inplace=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X = df_train
y = df_train.pop(14)
y_test = df_test.pop(14)
# dataset_id='adult_whole_data',
hdt.search(df_train, y, df_test, y_test, max_trials=3, batch_size=256, epochs=1, verbose=1, )
best_trial = hdt.get_best_trial()
assert best_trial

estimator = hdt.final_train(best_trial.space_sample, df_train, y)
score = estimator.predict(df_test)
result = estimator.evaluate(df_test, y_test)
print(result)
