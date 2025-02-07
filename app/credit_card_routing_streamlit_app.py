import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained PSP routing models
routing_data = joblib.load("models/routing_data_psp_models.joblib")

psp_models     = routing_data["models"]          # Dict of PSP -> tuned RandomForest model
label_encoders = routing_data["label_encoders"]  # Dict of LabelEncoder objects (or mappings)
bin_edges      = routing_data["bin_edges"]       # e.g. array([6, 72, 142, 270, 630])
bin_labels     = routing_data["bin_labels"]      # ["Low", "Medium", "High", "Very High"]
fee_success    = routing_data["fee_success"]     # e.g. {'Moneycard': 5, 'Goldcard':10,...}
fee_fail       = routing_data["fee_fail"]        # e.g. {'Moneycard': 2, 'Goldcard':5,...}
selected_feats = routing_data["selected_features"]# e.g. ['log_amount','3D_secured',...]


# Helper function to determine the "amount_category" from user input using the saved bin_edges and bin_labels
def get_amount_category(amount):
    """
    Use pd.cut with the loaded bin_edges, bin_labels to replicate
    the training-time qcut logic.
    """
    category = pd.cut([amount], bins=bin_edges, labels=bin_labels, include_lowest=True)[0]
    return str(category)  # "Low", "Medium", "High", or "Very High"


# Build the final feature vector for a single transaction & PSP, so we can predict success probability
def build_feature_vector(psp, country, card, is_3d_secure, amount):
    """
    Constructs the 10-feature vector (in training order) for one PSP.
    1) log_amount
    2) 3D_secured
    3) psp_card_le
    4) psp_3d_le
    5) card_3d_le
    6) psp_country_le
    7) card_country_le
    8) amount_category_Low
    9) amount_category_Medium
    10) amount_category_Very High
    """
    log_amt      = np.log1p(amount)
    three_d_val  = 1 if is_3d_secure else 0
    amt_cat      = get_amount_category(amount)
    cat_low      = 1 if amt_cat == "Low" else 0
    cat_medium   = 1 if amt_cat == "Medium" else 0
    cat_vhigh    = 1 if amt_cat == "Very High" else 0

    # Create the 5 interaction strings:
    psp_card_str      = f"{psp}_{card}"
    psp_3d_str        = f"{psp}_{three_d_val}"
    card_3d_str       = f"{card}_{three_d_val}"
    psp_country_str   = f"{psp}_{country}"
    card_country_str  = f"{card}_{country}"

    # Label-encode them
    # Fall back to 0 if unseen
    def safe_transform(labeler, val):
        if val in labeler.classes_:
            return labeler.transform([val])[0]
        else:
            return 0  # or some default

    pc_le  = safe_transform(label_encoders["psp_card"], psp_card_str)
    p3_le  = safe_transform(label_encoders["psp_3d"], psp_3d_str)
    c3_le  = safe_transform(label_encoders["card_3d"], card_3d_str)
    pco_le = safe_transform(label_encoders["psp_country"], psp_country_str)
    cco_le = safe_transform(label_encoders["card_country"], card_country_str)

    features = np.array([
        log_amt,
        three_d_val,
        pc_le,
        p3_le,
        c3_le,
        pco_le,
        cco_le,
        cat_low,
        cat_medium,
        cat_vhigh
    ]).reshape(1, -1)

    return features


# Single-Attempt Strategy
def single_attempt_decision(country, card, is_3d_secure, amount):
    """
    Pick the PSP with the lowest expected cost in a single attempt.
    Returns (best_psp, best_cost, best_prob).
    """
    best_psp = None
    best_cost = float('inf')
    best_prob = 0.0

    for psp, model in psp_models.items():
        fv = build_feature_vector(psp, country, card, is_3d_secure, amount)
        success_prob = model.predict_proba(fv)[0][1]

        # Expected cost = prob * fee_success + (1-prob)*fee_fail
        cost_s = fee_success[psp]
        cost_f = fee_fail[psp]
        ecost  = success_prob * cost_s + (1 - success_prob) * cost_f

        if ecost < best_cost:
            best_cost = ecost
            best_psp  = psp
            best_prob = success_prob

    return best_psp, best_cost, best_prob


# Multiple-Attempt Strategy
#  - We do repeated attempts until success or we reach 'max_attempts'.
#  - Each attempt picks the best PSP (lowest expected cost).
#  - We simulate success via a random draw (like your code does).
def multi_attempt_decision(country, card, is_3d_secure, amount, max_attempts=3):
    """
    Returns a dict with:
      - final_success (0/1)
      - total_fee
      - attempts_used
      - logs of each attempt if you want
    """
    attempts_used = 0
    total_fee     = 0.0
    success_flag  = 0  # 0 => fail, 1 => success

    attempt_logs = []  # optional detail

    while attempts_used < max_attempts and success_flag == 0:
        attempts_used += 1

        # 1) Find best PSP by expected cost (like single_attempt_decision)
        best_psp, best_cost, best_prob = single_attempt_decision(country, card, is_3d_secure, amount)

        # 2) Now randomly simulate success/failure
        #    e.g. if best_prob = 0.7, there's a 70% chance success, 30% fail
        random_draw    = np.random.rand()
        success_sim    = 1 if random_draw < best_prob else 0

        # 3) Determine cost (success or fail)
        cost = fee_success[best_psp] if success_sim == 1 else fee_fail[best_psp]
        total_fee += cost

        attempt_logs.append({
            'attempt': attempts_used,
            'chosen_psp': best_psp,
            'success_prob': best_prob,
            'random_draw': random_draw,
            'success_sim': success_sim,
            'cost': cost
        })

        # If success => we stop
        if success_sim == 1:
            success_flag = 1

    results = {
        'final_success': success_flag,
        'total_fee': total_fee,
        'attempts_used': attempts_used,
        'detail': attempt_logs
    }
    return results


# Build the Streamlit UI
st.title("Credit Card Routing App")

st.markdown("""
This app showcases **two** routing strategies:

1. **Single-Attempt**: 
   - We pick the PSP with the lowest expected cost one time and attempt the transaction.

2. **Multiple-Attempt**:
   - If the first attempt fails, we try again (up to a user-specified max) each time choosing
     the PSP that yields the lowest expected cost. We simulate success/fail with a random draw.

---
""")

# --- User Inputs ---
mode = st.radio("Select Routing Mode:", ["Single-Attempt", "Multiple-Attempt"])

country_choice = st.selectbox("Country", ["Germany", "Austria", "Switzerland"])
card_choice    = st.selectbox("Card Brand", ["Visa", "Diners", "Master"])
is_3d_choice   = (st.selectbox("3D Secure?", ["No", "Yes"]) == "Yes")
amount_choice  = st.slider("Transaction Amount (€)", min_value=6, max_value=630, value=100, step=1)

if mode == "Multiple-Attempt":
    max_attempts_choice = st.number_input("Max Attempts", min_value=1, max_value=10, value=3, step=1)

if st.button("Run Routing"):
    if mode == "Single-Attempt":
        best_psp, best_cost, best_prob = single_attempt_decision(
            country_choice, card_choice, is_3d_choice, amount_choice
        )
        st.subheader("Single-Attempt Results:")
        st.write(f"**Chosen PSP**: {best_psp}")
        st.write(f"**Predicted Success Probability**: {best_prob:.1%}")
        st.write(f"**Expected Cost**: €{best_cost:.2f}")

    else:
        # Multiple-Attempt
        outcome = multi_attempt_decision(
            country_choice, card_choice, is_3d_choice, amount_choice,
            max_attempts=int(max_attempts_choice)
        )
        st.subheader("Multiple-Attempts Results:")
        st.write(f"**Final Success**: {'Yes' if outcome['final_success'] == 1 else 'No'}")
        st.write(f"**Total Fee Paid**: €{outcome['total_fee']:.2f}")
        st.write(f"**Attempts Used**: {outcome['attempts_used']}")

        # If you want to show logs:
        with st.expander("See Attempt-by-Attempt Details"):
            for attempt_log in outcome['detail']:
                st.write(attempt_log)