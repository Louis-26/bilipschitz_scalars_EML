import os
from string import Template

def write_table(parameter, metric_hnn, metric_node):
    """
    parameter: a string of "layer_num-hidden_layer_num-learning_rate"
    metric: a list of metrics in the order of Train MSE, Test MSE, Test Rollout, Validation MSE
    """
    # parameter = "-".join(parameter)
    metric_hnn = " & ".join(metric_hnn)
    metric_node = " & ".join(metric_node)
    # Define the template string with $ for variable substitution
    template = Template(r"""adjusted parameter: $parameter\\
        \\ 
        \begingroup
        \setlength{\tabcolsep}{10pt}
        \renewcommand{\arraystretch}{1.5}
        \noindent
        \makebox[\textwidth][l]{%
            \begin{tabular}{|c|c|c|c|c|}\hline
            \diagbox{\centering Method}{\centering Metric} & Train MSE & Test MSE & Test Rollout & Validation MSE \\ \hline
            HNN  & $metric_hnn \\ \hline
            NODE & $metric_node \\ \hline
            \end{tabular}
        }
        \endgroup
        \\
        \\
        """)

    # Substitute the values into the template
    return template.substitute(
        parameter=parameter,
        metric_hnn=metric_hnn,
        metric_node=metric_node
    )

def read_data(f_name):
    output_dict=dict()
    file_dir="../rockfish_result/parameter_tuning/"+f_name
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("layer number-hidden layer number-learning rate:"):
                parameter = line.split(":")[1].strip()
            if line.startswith("Train"):
                line_li=line.split(":")
                metric=line_li[1].strip().split("-")
                output_dict[parameter]=metric
    return output_dict

def merge_dict(dict1, dict2):
    """
    two dictionaries may share the same keys, but each might have distinctive keys
    """
    new_dict=dict()
    for k,v in dict1.items():
        if k in dict2.keys():
            new_dict[k] = (v,dict2[k])
        else:
            new_dict[k] = (v,)
    for k,v in dict2.items():
        if k not in new_dict.keys():
            new_dict[k] = (v,)
    return new_dict

if __name__ == '__main__':
    # parameter = "3-100-0.01".split("-")
    # metric_hnn = "0.0001 & 0.0002 & 0.0003 & 0.0004".split(" & ")
    # metric_node = "0.0005 & 0.0006 & 0.0007 & 0.0008".split(" & ")
    # print(parameter, metric_hnn, metric_node)
    # print(write_table(parameter, metric_hnn, metric_node))
    hnn_metric_dict=read_data("parameter_tune_result_hnn.txt")
    node_metric_dict=read_data("parameter_tune_result_node.txt")
    merged_dict=merge_dict(hnn_metric_dict,node_metric_dict)
    output_file_name="result_as_table.txt"
    with open(output_file_name, "w") as f:
        pass
    for k,v in merged_dict.items():
        if len(v)==1:
            v=(v[0],["XX"]*4)
        args=(k,*v)
        # print(args)
        table_str=write_table(*args)
        with open(output_file_name, "a") as f:
            f.write(table_str)
            f.write("\n")

