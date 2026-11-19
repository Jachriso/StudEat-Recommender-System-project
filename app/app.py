"""StudEat Streamlit interface.

Run from the project root, with the API already running:
    python -m streamlit run app/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("STUDEAT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="StudEat", page_icon="🥡", layout="centered")
st.title("StudEat")
st.caption("Your weekly box of ready-made meals, matched to your tastes and budget.")


def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=5)


# --- API status -------------------------------------------------------------
try:
    health = api_get("/health").json()
    st.sidebar.success(f"API connected ({health['meals_in_catalogue']} meals)")
    options = api_get("/options").json()
except requests.exceptions.RequestException:
    st.sidebar.error(f"API not reachable at {API_URL}. Start it with: python -m uvicorn api.main:app --reload")
    st.stop()

if "user_id" in st.session_state:
    st.sidebar.info(f"Signed in as **{st.session_state['name']}** (user {st.session_state['user_id']})")

tab_signup, tab_box, tab_rate = st.tabs(["1. Sign up", "2. My weekly box", "3. Rate my meals"])

# --- Page 1: sign-up questionnaire -----------------------------------------
with tab_signup:
    with st.form("profile"):
        name = st.text_input("First name")
        diet = st.selectbox("Diet", options["diets"])
        allergies = st.multiselect("Allergies", options["allergens"])
        budget = st.slider("Weekly budget (EUR)", 20, 80, (35, 50))
        meals = st.number_input("Meals per week", min_value=1, max_value=14, value=10)
        submitted = st.form_submit_button("Create my profile")

    if submitted:
        if not name.strip():
            st.warning("Please enter your first name.")
        else:
            payload = {
                "name": name.strip(),
                "diet": diet,
                "allergies": allergies,
                "budget_min": budget[0],
                "budget_max": budget[1],
                "meals_per_week": int(meals),
            }
            r = requests.post(f"{API_URL}/profile", json=payload, timeout=5)
            if r.ok:
                st.session_state["user_id"] = r.json()["user_id"]
                st.session_state["name"] = name.strip()
                st.session_state.pop("box", None)
                st.success("Profile created! Open the « My weekly box » tab.")
            else:
                st.error(r.json().get("detail", "Could not create the profile."))

# --- Page 2: weekly box ----------------------------------------------------
with tab_box:
    if "user_id" not in st.session_state:
        st.info("Create your profile first.")
    else:
        if st.button("Get my box"):
            r = api_get(f"/recommend/{st.session_state['user_id']}")
            st.session_state["box"] = (r.ok, r.json())

        if "box" in st.session_state:
            ok, data = st.session_state["box"]
            if not ok:
                st.error(data.get("detail", "Could not get recommendations."))
            else:
                col1, col2 = st.columns(2)
                col1.metric("Meals", len(data["meals"]))
                col2.metric("Total", f"{data['total_price']:.2f} €")
                if not data["within_budget"]:
                    st.warning("This box is below your minimum budget.")

                df = pd.DataFrame(data["meals"])[["name", "cuisine", "price", "allergens"]]
                df["allergens"] = df["allergens"].apply(lambda a: ", ".join(a) if a else "-")
                df["price"] = df["price"].map(lambda p: f"{p:.2f} €")
                df.columns = ["Meal", "Cuisine", "Price", "Allergens"]
                st.dataframe(df, hide_index=True)
                st.caption(f"Recommendations: {data['model']}")


                
# --- Page 3: rate my meals -------------------------------------------------
with tab_rate:
    box = st.session_state.get("box")
    if not box or not box[0]:
        st.info("Get your weekly box first, then rate the meals you ate.")
    else:
        # One rating per distinct meal (a meal can appear twice in the box).
        meals = {m["id"]: m["name"] for m in box[1]["meals"]}
        st.write("How did you like your meals? Leave a meal empty if you did not eat it.")
        with st.form("ratings"):
            ratings = {}
            for meal_id, name in meals.items():
                st.caption(name)
                ratings[meal_id] = st.feedback("stars", key=f"rate_{meal_id}")
            sent = st.form_submit_button("Send my ratings")

        if sent:
            given = {mid: r + 1 for mid, r in ratings.items() if r is not None}  # stars widget returns 0-4
            if not given:
                st.warning("Rate at least one meal.")
            else:
                errors = 0
                for meal_id, rating in given.items():
                    r = requests.post(f"{API_URL}/log_interaction", timeout=5, json={
                        "user_id": st.session_state["user_id"], "meal_id": meal_id, "rating": rating})
                    errors += not r.ok
                if errors:
                    st.error(f"{errors} rating(s) could not be saved.")
                else:
                    st.success(f"Thanks! {len(given)} rating(s) saved.")
