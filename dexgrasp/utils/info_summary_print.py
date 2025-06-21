import yaml
import numpy as np
import os
def save_results_summary(results, filename, to_yaml=False):
    # output_data={}

    mean_succ_rate = float(np.mean(results['total_succ_rates']))
    results['total_succ_rates'] = mean_succ_rate
    print(f"Mean Success Rate: {mean_succ_rate:.4f}\n")
    # for key, val in results.items():
    #     if isinstance(val,list) and bool(1-isinstance(val[0],str)):
    #         results[key] = float(np.mean(val))

    # output_data = {
    #     # 'dataset_name': results.get('dataset_name', ''),
    #     'mean_total_succ_rate': round(mean_succ_rate, 4),
    #     'detail': results['detail']
    # }

    if to_yaml:
        os.makedirs("./results", exist_ok=True)
        filename = filename if filename.endswith('.yaml') else filename + '.yaml'
        with open("./results/{}".format(filename), 'w') as f:
            yaml.dump(results, f, allow_unicode=True)
            a=1
    else:
        filename = filename if filename.endswith('.txt') else filename + '.txt'
        with open("./results/{}".format(filename), 'w') as f:
            f.write(f"Dataset: {output_data['dataset_name']}\n")
            f.write(f"Mean Success Rate: {output_data['mean_total_succ_rate']:.4f}\n")
            f.write("Details:\n")
            for line in output_data['detail']:
                if isinstance(line, list):
                    f.write("  " + " | ".join(map(str, line)) + "\n")
                else:
                    f.write("  " + str(line) + "\n")
