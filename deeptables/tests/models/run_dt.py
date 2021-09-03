from sklearn.model_selection import train_test_split

from deeptables.datasets import dsutils
from deeptables.models import deeptable, deepnets

# loading data
df = dsutils.load_bank()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y = df_train.pop('y')
y_test = df_test.pop('y')

# training
config = deeptable.ModelConfig(nets=deepnets.DeepFM, earlystopping_patience=999,apply_class_weight=True)
dt = deeptable.DeepTable(config=config)
model, history = dt.fit(df_train, y, batch_size=32, epochs=5)

# evaluation
result = dt.evaluate(df_test, y_test, verbose=0)
print('score:', result)

# scoring
preds = dt.predict(df_test)

#
print('done')
