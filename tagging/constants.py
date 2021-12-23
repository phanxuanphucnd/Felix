"""Constants used by Felix Model."""

# Edit operations
KEEP = 'KEEP'
DELETE = 'DELETE'
PAD_TAG = 'PAD'

# Special tokens
PAD = '<pad>'
CLS = '<s>'
SEP = '</s>'
MASK = '<mask>'

# Special tokens that indicate the start and end of a span of deleted tokens
DELETE_SPAN_START = '<unused1>'
DELETE_SPAN_END = '<unused2>'

# For filtering out input tokens which are not used
DELETED_TAGS = frozenset([DELETE, PAD_TAG, PAD])

ID2TAGS = ["PAD",
           "KEEP",
           "DELETE",
           "KEEP|1",
           "KEEP|2",
           "KEEP|3",
        #    "KEEP|4",
        #    "KEEP|5",
        #    "KEEP|6",
        #    "KEEP|7",
        #    "KEEP|8",
        #    "KEEP|9",
        #    "KEEP|10"
           ]

TAGS2ID = {
    "PAD": 0,
    "KEEP": 1,
    "DELETE": 2,
    "KEEP|1": 3,
    "KEEP|2": 4,
    "KEEP|3": 5,
    # "KEEP|4": 6,
    # "KEEP|5": 7,
    # "KEEP|6": 8,
    # "KEEP|7": 9,
    # "KEEP|8": 10,
    # "KEEP|9": 11,
    # "KEEP|10": 12
}

TRAINING_FILE = "./data/wiki/train.conll"
VALID_FILE = "./data/wiki/valid.conll"