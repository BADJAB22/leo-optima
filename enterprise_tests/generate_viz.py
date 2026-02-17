import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations():
    if not os.path.exists('enterprise_stress_test_results.csv'):
        print("Results file not found.")
        return

    df = pd.DataFrame(pd.read_csv('enterprise_stress_test_results.csv'))
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Response Time Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration'], kde=True, color='skyblue')
    plt.title('Enterprise Load: Response Time Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.savefig('response_time_dist.png')
    plt.close()
    
    # 2. Routing Decisions
    plt.figure(figsize=(8, 8))
    df['route'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title('Routing Decision Distribution')
    plt.ylabel('')
    plt.savefig('routing_distribution.png')
    plt.close()
    
    # 3. Risk Assessment Accuracy
    plt.figure(figsize=(10, 6))
    risk_comparison = df.groupby(['expected_risk', 'actual_risk']).size().unstack(fill_value=0)
    sns.heatmap(risk_comparison, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Risk Assessment: Expected vs Actual')
    plt.savefig('risk_heatmap.png')
    plt.close()

    # 4. Trust Score Evolution (for the last few queries)
    # Note: trust_scores is stored as a string representation of a dict in CSV
    import ast
    last_query_trust = ast.literal_eval(df.iloc[-1]['trust_scores'])
    plt.figure(figsize=(10, 6))
    plt.bar(last_query_trust.keys(), last_query_trust.values(), color='teal')
    plt.title('Final Model Trust Scores after Adversarial Testing')
    plt.ylabel('Trust Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('final_trust_scores.png')
    plt.close()

    print("Visualizations generated successfully.")

if __name__ == "__main__":
    generate_visualizations()
