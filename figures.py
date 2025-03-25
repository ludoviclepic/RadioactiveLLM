import json
import matplotlib.pyplot as plt

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def extract_data(data):
    iterations = [entry['iteration'] for entry in data]
    p_values = [entry['p_value'] for entry in data]
    mean_r = [entry['mean_r'] for entry in data]
    log10_pvalues = [entry['log10_pvalue'] for entry in data]
    nb_disinct = [entry['nb_disinct'] for entry in data]
    nb = [entry['nb'] for entry in data]
    return iterations, p_values, mean_r, log10_pvalues, nb_disinct, nb

def plot_comparison(data1, data2, label1, label2):
    iterations1, p_values1, mean_r1, log10_pvalues1, nb_disinct1, nb1 = extract_data(data1)
    iterations2, p_values2, mean_r2, log10_pvalues2, nb_disinct2, nb2 = extract_data(data2)

    plt.figure(figsize=(8, 6))
    # Only plot log10_pvalues comparison
    plt.plot(iterations1, log10_pvalues1, label=label1)
    plt.plot(iterations2, log10_pvalues2, label=label2)
    plt.axhline(y=-1.3, color='r', linestyle='--', label='-1.3')
    plt.xlabel('Iteration')
    plt.ylabel('log10_pvalue')
    plt.title('log10_pvalue Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file1 = 'final_output_20/results_kgrams.jsonl'
    file2 = 'final_output_80/results_kgrams.jsonl'
    data1 = read_jsonl(file1)
    data2 = read_jsonl(file2)
    plot_comparison(data1, data2, 'Model 20%', 'Model 80%')