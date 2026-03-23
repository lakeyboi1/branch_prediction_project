class MultiBranchPredictor(BranchPredictor):
    type = 'MultiBranchPredictor'
    cxx_class = 'gem5::branch_prediction::MultiBranchPredictor'
    cxx_header = "cpu/pred/multi_branch.hh"
    local_table_size    = Param.Unsigned(2048, "Local predictor table entries")
    global_table_size   = Param.Unsigned(8192, "Global predictor table entries")
    global_history_bits = Param.Unsigned(13,   "Global history register bits")
    choice_table_size   = Param.Unsigned(8192, "Choice predictor table entries")