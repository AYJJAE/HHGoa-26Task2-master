import json
with open('data/mock_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = []
for i in range(5):
    for d in data:
        new_d = d.copy()
        new_d['query_id'] = str(d.get('query_id', '')) + f'_{i}'
        new_data.append(new_d)
with open('data/mock_dataset_100.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
print('Created 100 queries')
