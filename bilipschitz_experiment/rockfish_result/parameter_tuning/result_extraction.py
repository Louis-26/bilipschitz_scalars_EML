import os
from string import Template

from sympy.codegen.ast import String


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


def write_table_combined(method, metric_hnn, metric_node):
    template = Template(r"""\begin{table}[ht]
        \centering
        \caption*{$method}
        \begingroup
        \setlength{\tabcolsep}{10pt}
        \renewcommand{\arraystretch}{1.5}
        \makebox[\textwidth][l]{
            \begin{tabular}{|c|c|c|c|c|}\hline
            \diagbox{\centering parameters}{\centering Metric} & Train MSE & Test MSE & Test Rollout & Validation MSE \\ \hline
            3-100-0.01  & XXX & XXX & XXX & XXX \\ \hline
            NODE & XXX & XXX & XXX & XXX \\ \hline
            \end{tabular}
        }
        \endgroup"""
                        )


def read_data(f_name, bilipschitz=False):
    output_dict = dict()
    prefix = "parameter_tune_no_embedding/" if not bilipschitz else "parameter_tune_with_embedding/"
    file_dir = prefix + f_name
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("layer number-hidden layer number-learning rate:"):
                parameter = line.split(":")[1].strip()
            if line.startswith("Train"):
                line_li = line.split(":")
                metric = line_li[1].strip().split("-")
                output_dict[parameter] = metric
    return output_dict


def merge_dict(dict1, dict2):
    """
    two dictionaries may share the same keys, but each might have distinctive keys
    """
    new_dict = dict()
    for k, v in dict1.items():
        if k in dict2.keys():
            new_dict[k] = (v, dict2[k])
        else:
            new_dict[k] = (v,)
    for k, v in dict2.items():
        if k not in new_dict.keys():
            new_dict[k] = (v,)
    return new_dict


# it outputs 27 tables, each of them contains the comparison of HNN and NODE for a specific hyperparameter setting
def write_table_all(output_file_name, bilipschitz):
    hnn_metric_dict = read_data("parameter_tune_result_hnn.txt", bilipschitz)
    node_metric_dict = read_data("parameter_tune_result_node.txt", bilipschitz)
    merged_dict = merge_dict(hnn_metric_dict, node_metric_dict)
    # output_file_name="result_as_table_original.txt"
    # output_file_name = "result_as_table_bilipschitz.txt"
    with open(output_file_name, "w") as f:
        pass
    for k, v in merged_dict.items():
        if len(v) == 1:
            v = (v[0], ["XX"] * 4)
        args = (k, *v)
        # print(args)
        table_str = write_table(*args)
        with open(output_file_name, "a") as f:
            f.write(table_str)
            f.write("\n")


# it outputs 2 tables(HNN and NODE), each of them contains the comparison of all hyperparameter settings given the method
def write_table_comb(output_file_name, bilipschitz):
    hnn_metric_dict = read_data("parameter_tune_result_hnn.txt", bilipschitz)
    node_metric_dict = read_data("parameter_tune_result_node.txt", bilipschitz)
    merged_dict = merge_dict(hnn_metric_dict, node_metric_dict)
    caption_suffix=" with bilipschitz embedding" if bilipschitz else " without bilipschitz embedding"
    with open(output_file_name, "w") as f:
        pass
    for method in ["HNN", "NODE"]:
        if method == "HNN":
            metric_dict = hnn_metric_dict
        else:
            metric_dict = node_metric_dict

        with open(output_file_name, "a") as f:
            str1 = Template(r"""\begin{table}[ht]
                \centering
                \caption*{$method}
                \begingroup
                \setlength{\tabcolsep}{10pt}
                \renewcommand{\arraystretch}{1.5}
                \makebox[\textwidth][l]{
                    \begin{tabular}{|c|c|c|c|c|}\hline
                    \diagbox{\centering parameters}{\centering Metric} & Train MSE & Test MSE & Test Rollout & Validation MSE \\ \hline""").substitute(
                method=method+caption_suffix)
            f.write(str1 + "\n")
            for k, v in metric_dict.items():
                f.write(f"{k} & "+" & ".join(v) + " \\\\ \\hline\n")
            str2=r"""\end{tabular}
        }
        \endgroup
        \end{table}"""
            f.write(str2 + "\n")

if __name__ == '__main__':
    write_table_all(output_file_name="result_as_table_original.txt", bilipschitz=False)
    write_table_all(output_file_name="result_as_table_bilipschitz.txt", bilipschitz=True)
    write_table_comb(output_file_name="comb_table_original.txt", bilipschitz=False)
    write_table_comb(output_file_name="comb_table_bilipschitz.txt", bilipschitz=True)
