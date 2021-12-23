import os
import json

def create_label_map(max_mask=10, use_pointing=True, save_file='ext/label_map.json'):
    label_map = {
        'PAD': 0, 
        'KEEP': 1, 
        'DELETE': 2, 
        # 'SWAP': 3
    }

    # Create Insert 1 MASK to Intert n MASKS
    for i in range(0, max_mask):
        label_map[f'KEEP|{i+1}'] = len(label_map)

        if not use_pointing:
            label_map[f'DELETE|{i+1}'] = len(label_map)

    out_dir = '/'.join(save_file.split('/')[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False)

    print(f'Created new label map with {len(label_map)} labels. Saved to `{save_file}`.')

create_label_map()
