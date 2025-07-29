from string import Template

def write_table(parameter, metric_hnn, metric_node):
    """
    parameter: a list of (layer_num, hidden_layer_num, learning_rate)
    metric: a list of metrics in the order of Train MSE, Test MSE, Test Rollout, Validation MSE
    """
    parameter = "-".join(parameter)
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








with open("../rockfish_result/parameter_tuning/parameter_tune_result.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("Train"):
            line_li=line.split(":")
            metric=line_li[1].strip().split("-")
            # print(" & ".join(metric))


if __name__ == '__main__':
    parameter = "3-100-0.01".split("-")
    metric_hnn = "0.0001 & 0.0002 & 0.0003 & 0.0004".split(" & ")
    metric_node = "0.0005 & 0.0006 & 0.0007 & 0.0008".split(" & ")
    print(parameter, metric_hnn, metric_node)
    print(write_table(parameter, metric_hnn, metric_node))