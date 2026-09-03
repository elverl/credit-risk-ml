"""
Project configuration for the credit-risk modeling pipeline.
"""

RANDOM_STATE = 42
TARGET = "incumplimiento"

FEATURES = [
    "n_atr_1d_6",
    "cl_9_1.0",
    "cl_12_1.0",
    "cl_18_2.0",
    "cl_9_3.0",
    "ind_vjrc_36_1",
    "ind_per_x9_1.0",
    "prc_u_tc_cns",
    "tas_u_tc",
    "c_tc_mn_24",
    "c_tc_mn_24_sld",
    "c_pcns",
    "vr_s_t_36",
    "mx_ltc_36",
    "ipc_t6",
]