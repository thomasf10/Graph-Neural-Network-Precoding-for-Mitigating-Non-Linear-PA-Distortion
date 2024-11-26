

ACT2NAME = {

    None: "linear_activation",
    "sigmoid": "sigmoid_activation",
    "silu": "silu_activation",
    "swish": "swish_activation",
    "tanh": "tanh_activation",
    "relu": "relu_activation",
    "gelu": "gelu_activation",
    "selu": "selu_activation",
    "lrelu": "lrelu_activation",
    "prelu": "prelu_activation",
    "elu": "elu_activation",

}

SKIP2NAME = {
    None: "",
    "gnn": '_skip_gnn_gating',
    "mlp": '_skip_mlp_gating',
    "learned_per_layer": '_skip_fixed_gating'
}


def get_name(layer_name, nr=0, activation_string=None, skip=None):

    if activation_string in ACT2NAME and skip in SKIP2NAME:
        name = f"{layer_name}_{nr}_{ACT2NAME[activation_string]}{SKIP2NAME[skip]}"
        return name
    elif activation_string not in ACT2NAME:
        raise KeyError(f"key {activation_string} not found in ACT2NAME mapping {list(ACT2NAME.keys())}")
    elif skip not in SKIP2NAME.keys():
        raise KeyError(f"key {skip} not found in SKIP2NAME mapping {list(SKIP2NAME.keys())}")
    else:
        print("something weird went wrong check naming.py")
