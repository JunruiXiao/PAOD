import mmcv

data_path = '/home/xiaojunrui/data/crowdhuman/annotations/annotation_val.json'
data = mmcv.load(data_path)

print(data['annotations'][0])
print(data['images'][0])
print(data['categories'])