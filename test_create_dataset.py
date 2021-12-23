import transformers
from masked_lm.pointing_converter import PointingConverter
from masked_lm.insertion_converter import InsertionConverter
from masked_lm.datasets.insertion_dataset import InsertionDataset

do_lower_case = True
use_open_vocab = True
max_seq_length = 128
max_predictions_per_seq = 20

point_converter = PointingConverter({}, do_lower_case=do_lower_case)

insertion_converter = InsertionConverter(
    max_seq_length=max_seq_length,
    max_predictions_per_seq=max_predictions_per_seq,
    label_map_file='./ext/label_map.json',
    lm_pretrained_path='./models/BDIRoBerta/',
    fall_back_mode='augment'
)

dataset = InsertionDataset(
    data_path='./data/ttrain.csv', 
    lm_pretrained_path='./models/BDIRoBerta',
    label_map_file='./ext/label_map.json',
    point_converter=point_converter,
    insertion_converter=insertion_converter,
    use_open_vocab=use_open_vocab,
    max_seq_length=max_seq_length,
    do_lower_case=do_lower_case
)

print(len(dataset))