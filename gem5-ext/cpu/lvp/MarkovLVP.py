from m5.params import *
from m5.SimObject import SimObject

class MarkovLVP(SimObject):
    type       = 'MarkovLVP'
    cxx_class  = 'gem5::MarkovLVP'
    cxx_header = 'cpu/lvp/markov_lvp.hh'

    table_size      = Param.Unsigned(4096,  "Markov table entries (power of 2)")
    context_depth   = Param.Unsigned(1,     "Value history depth (1 = last-value predictor)")
    conf_threshold  = Param.Unsigned(2,     "Min confidence counter to issue prediction")
    max_conf        = Param.Unsigned(7,     "Confidence counter saturation ceiling")