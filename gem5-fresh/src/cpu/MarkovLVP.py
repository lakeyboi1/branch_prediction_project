from m5.params import *
from m5.SimObject import SimObject

class MarkovLVP(SimObject):
    type       = 'MarkovLVP'
    cxx_class  = 'gem5::MarkovLVP'
    cxx_header = 'cpu/markov_lvp.hh'

    table_size     = Param.Unsigned(4096,
        "Markov table entries (power of 2)")
    context_depth  = Param.Unsigned(1,
        "Value history depth — number of past strides used as context")
    conf_threshold = Param.Unsigned(1,
        "Min confidence counter value before issuing a prediction (0-15)")
    max_conf       = Param.Unsigned(15,
        "Saturating counter ceiling (4-bit = max 15)")