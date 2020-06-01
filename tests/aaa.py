# -*- encoding: utf-8 -*-
from deeptables.models import deeptable, ModelConfig
import pickle as pkl

nets=['dnn_nets','linear','cin_nets','fm_nets','afm_nets','opnn_nets','ipnn_nets','pnn_nets','cross_nets','cross_dnn_nets','dcn_nets','autoint_nets','fg_nets','fgcnn_cin_nets','fgcnn_fm_nets','fgcnn_ipnn_nets','fgcnn_dnn_nets','fibi_nets','fibi_dnn_nets']
config = ModelConfig(nets=nets)

with open('/Users/wuhf/Downloads/model_config.pkl', 'rb') as f:
    pass
    # config  = pkl.load(f)
dt = deeptable.DeepTable(config=config)

print(dt)

with open('/Users/wuhf/Downloads/7c4373fe6880477185d4bb0674f99ba2_1.pkl', 'rb') as f:
    import pickle as pkl
    df = pkl.load(f)
X_train = df
y_trian = df.pop('y')

dt.fit(X_train, y_trian)



