import streamlit as st
import numpy as np
import joblib
import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.feature_engineering import PokerFeatureEngine

# Page config
st.set_page_config(
    page_title="Poker Hand Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    engine = PokerFeatureEngine()
    model_loaded = True
except Exception as e:
    st.error("Error loading the classification model. Please ensure the model files are present.")
    st.code(f"{type(e).__name__}: {str(e)}")
    model_loaded = False

# Header
st.title("♠️♥️ Poker Hand Classifier ♦️♣️")
st.markdown("""
Welcome to the **Poker Hand Classifier**! This tool automatically identifies what poker hand you have based on your 5 cards.

**How it works:** Select 5 cards below, and the classifier will instantly tell you what poker hand they form.
""")

st.divider()

# Instructions in expander
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Instructions:
    
    1. **Select 5 cards** using the dropdowns below (choose suit and rank for each card)
    2. Make sure **all 5 cards are different** (no duplicates allowed in poker)
    3. Click the **"Classify Hand"** button
    4. The app will show you:
       - What poker hand you have
       - How confident the model is
       - Technical features used for classification (optional)
    
    ### Rules:
    - You must select exactly 5 cards
    - Each card can only appear once (standard 52-card deck)
    - The app evaluates standard poker hand rankings
    """)

# Card selection interface
st.subheader("🎴 Select Your 5 Cards")

suits = {'Hearts': 1, 'Spades': 2, 'Diamonds': 3, 'Clubs': 4}
suit_symbols = {1: '♥️', 2: '♠️', 3: '♦️', 4: '♣️'}
ranks = {
    'Ace': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13
}
rank_display = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}

cols = st.columns(5)
cards = []

for i, col in enumerate(cols):
    with col:
        st.write(f"**Card {i+1}**")
        suit = st.selectbox(
            "Suit", 
            list(suits.keys()), 
            key=f"suit_{i}", 
            label_visibility="collapsed"
        )
        rank = st.selectbox(
            "Rank", 
            list(ranks.keys()), 
            key=f"rank_{i}", 
            label_visibility="collapsed"
        )
        cards.append((suits[suit], ranks[rank]))

# Validation: check for duplicate cards
def validate_cards(cards):
    """Check if there are any duplicate cards (same suit AND rank)"""
    seen = set()
    duplicates = []
    
    for suit, rank in cards:
        card = (suit, rank)
        if card in seen:
            rank_name = rank_display.get(rank, str(rank))
            suit_name = [k for k, v in suits.items() if v == suit][0]
            duplicates.append(f"{rank_name}{suit_symbols[suit]} ({suit_name})")
        seen.add(card)
    
    return duplicates

# Check for duplicates
duplicates = validate_cards(cards)
has_duplicates = len(duplicates) > 0

# Display selected cards visually
st.markdown("### Your Hand:")
card_cols = st.columns(5)
for i, (suit, rank) in enumerate(cards):
    with card_cols[i]:
        rank_name = rank_display.get(rank, str(rank))
        st.markdown(
            f"<h1 style='text-align: center; font-size: 4em;'>{rank_name}{suit_symbols[suit]}</h1>", 
            unsafe_allow_html=True
        )

st.divider()

# Show error if duplicates exist
if has_duplicates:
    st.error(f"⚠️ **Invalid hand!** You selected the same card multiple times: {', '.join(duplicates)}")
    st.warning("In poker, each card can only appear once. Please select 5 different cards to classify your hand.")

# Predict button - centered and prominent
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_button = st.button(
        "🎯 Classify Hand", 
        type="primary", 
        use_container_width=True,
        disabled=has_duplicates or not model_loaded
    )

# Only process if button clicked and no duplicates
if classify_button and not has_duplicates and model_loaded:
    
    # Create input array matching training data format
    input_row = []
    for suit, rank in cards:
        input_row.extend([suit, rank])
    
    # Convert to pandas Series for feature engineering
    import pandas as pd
    input_series = pd.Series(input_row)
    
    # Extract features
    features_dict = engine.transform_row(input_series)
    features_array = np.array([list(features_dict.values())])
    
    # Scale and predict
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Display results prominently
    st.markdown("---")
    
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        st.success(f"# 🎉 {PokerFeatureEngine.get_hand_name(prediction)}")
        
        # Add description of the hand
        hand_descriptions = {
            0: "No matching cards. This is the lowest-ranking hand in poker.",
            1: "Two cards of the same rank. A common hand in poker.",
            2: "Two different pairs in one hand. Better than one pair!",
            3: "Three cards of the same rank. A strong hand.",
            4: "Five cards in sequential rank order (mixed suits). A very good hand!",
            5: "Five cards of the same suit (not in sequence). A powerful hand!",
            6: "Three of a kind plus a pair. An excellent hand!",
            7: "Four cards of the same rank. An extremely rare and strong hand!",
            8: "Five cards in sequence, all of the same suit. A phenomenal hand!",
            9: "10-J-Q-K-A all of the same suit. The best possible hand in poker!"
        }
        
        st.info(hand_descriptions[prediction])
    
    with result_col2:
        # Confidence gauge
        confidence = probabilities[prediction] * 100
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.1f}%",
            help="How certain the model is about this classification"
        )
    
    st.divider()
    
    # Technical details in expander
    with st.expander("🔍 Technical Details (for data enthusiasts)"):
        st.subheader("Feature Analysis")
        st.caption("These are the engineered features the model uses to make predictions:")
        
        # First row - primary boolean features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Is Flush", 
                "Yes" if features_dict['is_flush'] else "No",
                help="All cards have the same suit"
            )
        with col2:
            st.metric(
                "Is Straight", 
                "Yes" if features_dict['is_straight'] else "No",
                help="Cards form a sequential rank pattern"
            )
        with col3:
            st.metric(
                "Is Royal Straight", 
                "Yes" if features_dict['is_royal_straight'] else "No",
                help="Specifically 10-J-Q-K-A sequence"
            )
        
        # Second row - rank pattern features
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(
                "Unique Ranks", 
                features_dict['unique_ranks'],
                help="Number of different card ranks"
            )
        with col5:
            st.metric(
                "Max Rank Count", 
                features_dict['max_rank_count'],
                help="Most frequent rank occurrence"
            )
        with col6:
            st.metric(
                "Second Max Count", 
                features_dict['second_max_rank_count'],
                help="Second most frequent rank"
            )
        
        # Probability distribution
        st.subheader("Prediction Probabilities")
        st.caption("The model's confidence for each possible hand type:")
        
        prob_df = pd.DataFrame({
            'Hand': [PokerFeatureEngine.get_hand_name(i) for i in range(10)],
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Hand'), height=300)

# Sidebar with educational content
st.sidebar.header("📚 Poker Hand Rankings")
st.sidebar.markdown("""
Hands are ranked from best to worst:
""")

hand_rankings = [
    ("9. Royal Flush", "10-J-Q-K-A, all same suit", "🏆"),
    ("8. Straight Flush", "Five sequential cards, same suit", "⭐"),
    ("7. Four of a Kind", "Four cards of same rank", "💎"),
    ("6. Full House", "Three of a kind + pair", "🎯"),
    ("5. Flush", "Five cards, all same suit", "✨"),
    ("4. Straight", "Five sequential cards", "📊"),
    ("3. Three of a Kind", "Three cards of same rank", "🎲"),
    ("2. Two Pairs", "Two different pairs", "👥"),
    ("1. One Pair", "Two cards of same rank", "🎴"),
    ("0. Nothing", "No matching cards", "🃏")
]

for rank, description, emoji in hand_rankings:
    st.sidebar.markdown(f"**{emoji} {rank}**")
    st.sidebar.caption(description)
    st.sidebar.divider()

# Sidebar examples
st.sidebar.header("💡 Example Hands")

example_hands = {
    "Royal Flush": "10♠ J♠ Q♠ K♠ A♠",
    "Straight Flush": "5♣ 6♣ 7♣ 8♣ 9♣",
    "Four of a Kind": "A♥ A♠ A♦ A♣ K♠",
    "Full House": "K♥ K♠ K♦ 10♥ 10♠",
    "Flush": "2♦ 5♦ 8♦ J♦ K♦",
    "Straight": "5♥ 6♠ 7♦ 8♣ 9♥"
}

selected_example = st.sidebar.selectbox(
    "View example:",
    list(example_hands.keys())
)

st.sidebar.code(example_hands[selected_example])
st.sidebar.caption("Try creating this hand above to see the classification!")

# Footer
st.sidebar.divider()
st.sidebar.header("ℹ️ About This Tool")
st.sidebar.markdown("""
This classifier uses machine learning to identify poker hands with 100% accuracy.

**How it works:**
The model analyzes patterns in your cards (like matching suits or sequential ranks) and classifies them into one of 10 standard poker hands.

**Accuracy:** Tested on over 1 million poker hands with perfect results.
""")

# Technical info in expander
with st.sidebar.expander("🔬 Technical Information"):
    st.markdown("""
    **Model:** Logistic Regression
    
    **Features:** 6 domain-engineered features
    - Flush detection
    - Straight detection
    - Royal straight detection
    - Rank pattern analysis
    
    **Performance:** 100% accuracy on 1M+ test samples
    
    This demonstrates that thoughtful feature engineering can outperform complex models on well-defined problems.
    """)