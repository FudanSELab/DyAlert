import datetime
import os
import random
import numpy as np
import json
from scipy.stats import pearsonr
from sklearn import metrics
from fastdtw import fastdtw
from tqdm import tqdm

business_units =["Online Shopping", "Delivery Service", "Order Home"]

# metric key, time of duration, detailed rule, rule type (if type=="minute on minute")
rules = [
    ["success rate", "in the last 2 minutes", "was down", "minute on minute"],
    ["success count", "in the last 5 minutes", "was continuously less than"],
    ["failure rate", "in the last 1 minutes", "was over", "minute on minute"],
    ["failure count", "at the current period", "was continuously more than"]
]


def calculate_metrics_correlation(data, method):
    correlation_matrix = np.zeros((data.shape[0], data.shape[0]))
    if method == 'dtw':
        for i in tqdm(range(data.shape[0])):
            if data[i, :].max() == data[i, :].min():
                data[i, :] = data[i, :] - data[i, :].min()
            else:
                data[i, :] = (data[i, :] - data[i, :].min()) / (data[i, :].max() - data[i, :].min())
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            if method == 'pearson':
                if i == j:
                    correlation_degree = 1
                else:
                    correlation_degree = pearsonr(data[i, :], data[j, :])[0]
            elif method == 'mutual information':
                correlation_degree = metrics.normalized_mutual_info_score(data[i, :], data[j, :])
            elif method == 'dtw':
                if i != j:
                    if data[i, :].max() == data[i, :].min() or data[j, :].max() == data[j, :].min():
                        correlation_degree = np.nan
                    else:
                        correlation_degree, _ = fastdtw(data[i, :], data[j, :])
                else:
                    correlation_degree = 0
            else:
                print('Error Method!')
                return None
            correlation_matrix[i, j] = correlation_degree
            correlation_matrix[j, i] = correlation_matrix[i, j]

    correlation_matrix = np.ma.masked_invalid(correlation_matrix)
    # The difference calculated by dtw should be normalized to 0~1
    if method == 'dtw':
        if np.nanmax(correlation_matrix) - np.nanmin(correlation_matrix) != 0:
            correlation_matrix = (correlation_matrix - np.nanmin(correlation_matrix)) / (
                        np.nanmax(correlation_matrix) - np.nanmin(correlation_matrix))
        else:
            correlation_matrix = (correlation_matrix - np.nanmean(correlation_matrix))
        correlation_matrix = 1 - correlation_matrix
    return correlation_matrix


def generate_data(total_num, metric_len, correlation_method):
    snapshots = generate_snapshots(total_num, metric_len, correlation_method)
    os.makedirs(f'./data/metric_{metric_len}', exist_ok=True)
    idx = 0
    while (idx*50)+1 <= total_num:
        with open(f'./data/metric_{metric_len}/snapshots_{(idx*50)+1}-{(idx+1)*50 if (idx+1)*50 <= total_num else total_num}.json', 'w') as f:
            json.dump(snapshots[idx*50: (idx+1)*50], f)
        idx += 1


def generate_snapshots(total_num, metric_len, correlation_method):
    alert_id = 0
    snapshots = []
    t = datetime.datetime.strptime("2022-07-01 00:00", "%Y-%m-%d %H:%M")
    # init_metric (the total number of metric: 100, the number of data points in each metric: metric_len)
    metric_names = ["metric_" + str(x) for x in range(100)]
    metric_x = []
    for i in range(100):
        metric_x.append(np.random.rand(metric_len).tolist())
    print('Start generating snapshots:')
    for snapshot_i in tqdm(range(total_num)):
        time_interval = random.randint(1, 10)
        t += datetime.timedelta(minutes=time_interval)
        snapshot = {"alert_names": [], "messages": [], "metric_names": metric_names,
                    "metric_x": [], "alert_from_metric_edges": [[], []], "alert_link_alert_edges": [[], []],
                    "alert_is_new": [], "time": t.strftime("%Y-%m-%d %H:%M")}
        if snapshot_i > 0:
            # generate historical data
            last_snapshot = snapshots[-1]
            remain_num = random.randint(0, len(last_snapshot["alert_names"]))
            remainder = random.sample([x for x in range(len(last_snapshot["alert_names"]))], remain_num)
            snapshot["alert_names"].extend([last_snapshot["alert_names"][x] for x in range(len(last_snapshot["alert_names"])) if x in remainder])
            snapshot["messages"].extend([last_snapshot["messages"][x] for x in range(len(last_snapshot["messages"])) if x in remainder])
            snapshot["alert_is_new"].extend([0] * remain_num)
            snapshot["metric_x"] = [x[time_interval:] for x in last_snapshot["metric_x"]]
        else:
            snapshot["metric_x"] = metric_x
        # generate new alert
        alert_num = random.randint(1, 5)
        for alert_i in range(alert_num):
            # generate alert name (auto-increment ids)
            snapshot["alert_names"].append('alert_' + str(alert_id))
            alert_id += 1
            # generate messages
            message = f'[{t.strftime("%Y-%m-%d %H:%M")} {random.choice(business_units)}] '
            rule_id = random.choice([x for x in range(len(rules))])
            if rule_id % 2 == 0:
                current_value = random.randint(10, 90)
                message += f'The {rules[rule_id][0]} [Current value is {current_value}%] {rules[rule_id][1]} {rules[rule_id][2]} '
                if rule_id // 2 == 0:
                    message += f'{random.randint(current_value+1, 100)}%'
                else:
                    message += f'{random.randint(0, current_value-1)}%'
                message += rules[rule_id][3]
            else:
                current_value = random.randint(10, 10000)
                message += f'The {rules[rule_id][0]} [Current value is {current_value}%] {rules[rule_id][1]} {rules[rule_id][2]} '
                if rule_id // 2 == 0:
                    message += f'{random.randint(current_value + 1, 10100)}%'
                else:
                    message += f'{random.randint(0, current_value - 1)}%'
            snapshot["messages"].append(message)
            # alert_is_new
            snapshot["alert_is_new"].append(1)
            # generate edges
            related_metric_num = random.randint(1, 3)
            related_metrics = random.sample([x for x in range(100)], related_metric_num)
            snapshot["alert_from_metric_edges"][0].extend([alert_i] * related_metric_num)
            snapshot["alert_from_metric_edges"][1].extend(related_metrics)

            linked_alert_num = random.randint(0, 3 if alert_num > 3 else alert_num)
            if linked_alert_num > 0:
                linked_alerts = random.sample([x for x in range(alert_num)], linked_alert_num)
                snapshot["alert_link_alert_edges"][0].extend([alert_i] * linked_alert_num)
                snapshot["alert_link_alert_edges"][1].extend(linked_alerts)
        # generate new metric data points
        for metric_i in range(100):
            snapshot["metric_x"][metric_i].extend(np.random.rand(time_interval).tolist())
        snapshots.append(snapshot)
        # calculate pearson coefficient (which can be accelerated by multiprocessing)
        edge_index = [[], []]
        correlation_matrix = calculate_metrics_correlation(np.asarray(snapshot['metric_x']), correlation_method)
        edge_weight = []
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                if correlation_matrix._mask[i, j] == False and i != j:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_weight.append(correlation_matrix[i][j])
        snapshot['metric_correlation_metric_edges_weight'] = edge_weight
        snapshot['metric_correlation_metric_edges'] = edge_index
    return snapshots


if __name__ == '__main__':
    # We can not make our experiment data publicly available since it is confidential data from a commercial enterprise.

    # Thus, to show the format of the json data for dynamic graph construction,
    # we write this function for generating example json data randomly,
    # whose results do not correspond to the experimental results in our paper.
    generate_data(total_num=65, metric_len=30, correlation_method='pearson')