import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_poker_data(train_path, test_path):
    """
    Load poker datasets. First row contains headers.
    """
    # Load with headers (pandas automatically uses first row as column names)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Rename to shorter, code-friendly names
    new_columns = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'hand']
    train_df.columns = new_columns
    test_df.columns = new_columns
    
    print(f"Training set: {len(train_df):,} hands")
    print(f"Test set: {len(test_df):,} hands")
    
    return train_df, test_df


def analyze_class_distribution(df, hand_col='hand'):
    """
    The first thing to check. This will reveal the massive imbalance
    that makes this problem interesting.
    """
    from src.feature_engineering import PokerFeatureEngine
    
    counts = df[hand_col].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(3)
    
    print("\nClass Distribution:")
    print("-" * 60)
    for hand_id in sorted(counts.index):
        hand_name = PokerFeatureEngine.get_hand_name(hand_id)
        count = counts[hand_id]
        pct = percentages[hand_id]
        print(f"{hand_id}: {hand_name:20s} | {count:8,} ({pct:6.3f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax1 = axes[0]
    counts.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Class Distribution (Raw Counts)')
    ax1.set_xlabel('Poker Hand')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels([PokerFeatureEngine.get_hand_name(i) for i in counts.index], 
                        rotation=45, ha='right')
    
    # Log scale to actually see rare classes
    ax2 = axes[1]
    counts.plot(kind='bar', ax=ax2, color='coral', logy=True)
    ax2.set_title('Class Distribution (Log Scale)')
    ax2.set_xlabel('Poker Hand')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_xticklabels([PokerFeatureEngine.get_hand_name(i) for i in counts.index], 
                        rotation=45, ha='right')
    
    plt.tight_layout()
    return fig