"""StudEat Streamlit interface.

Run from the project root, with the API already running:
    python -m streamlit run app/app.py

Visual principles (see the design canvas): equal diet badges listed A to Z with no
religious symbol, a neutral lunch-box mascot, food rather than people, allergens
shown with words and not only colours, contrast-safe colours (#C2410C on white).
"""

import base64
import html
import os
from collections import Counter
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("STUDEAT_API_URL", "http://localhost:8000")
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = Image.open(ASSETS / "logo.png")

st.set_page_config(page_title="StudEat", page_icon=LOGO.crop((0, 0, LOGO.height, LOGO.height)), layout="wide")

# --- Look and feel -----------------------------------------------------------
# Colours come from .streamlit/config.toml; this CSS only adds what the theme cannot do.
st.html("""
<style>
.block-container {max-width: 1150px; padding-top: 4.5rem;}
h1.se-title {margin: 0; font-size: 48px; font-weight: 800; letter-spacing: -1px; line-height: 1.1;}
h1.se-title span {color: #C2410C;}
.se-muted {color: #6B5A4C;}
.se-tagline {margin: 4px 0 0; font-size: 18px; color: #6B5A4C;}
.se-bubble {position: relative; background: #FFFFFF; border-radius: 20px; padding: 16px 18px;
  font-size: 15px; line-height: 1.5; box-shadow: 0 4px 14px rgba(194, 65, 12, 0.10); margin-bottom: 18px;}
.se-bubble:after {content: ""; position: absolute; left: 44px; bottom: -10px; width: 20px; height: 20px;
  background: #FFFFFF; transform: rotate(45deg);}
.se-note {background: #FFFFFF; border-radius: 16px; padding: 14px; font-size: 13px; line-height: 1.5; color: #4A3B2F;}
.se-header {display: flex; align-items: center; gap: 20px;}
.se-header h2 {margin: 0; padding: 0; font-size: 32px; font-weight: 800; line-height: 1.2;}
.se-header p {margin: 4px 0 0; font-size: 16px; color: #6B5A4C;}
.se-panel {background: #FFFFFF; border: 1px solid #FFE3D3; border-radius: 20px; padding: 18px 24px; margin: 8px 0 4px;}
.se-bar {height: 12px; border-radius: 999px; background: #FFE8DC; overflow: hidden; margin-top: 10px;}
.se-bar > div {height: 100%; background: #C2410C; border-radius: 999px;}
.se-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px;}
.se-card {background: #FFFFFF; border: 1px solid #FFE3D3; border-radius: 20px; padding: 20px;
  display: flex; flex-direction: column; gap: 10px;}
.se-card h3 {margin: 0; padding: 0; font-size: 17px; font-weight: 600; line-height: 1.35;}
.se-row {display: flex; justify-content: space-between; align-items: center;}
.se-price {background: #C2410C; color: #FFFFFF; border-radius: 999px; padding: 2px 12px; font-size: 14px; font-weight: 600;}
.se-chips {display: flex; flex-wrap: wrap; gap: 6px;}
.se-ok {display: flex; align-items: center; gap: 4px; background: #E6F5EC; color: #1F6B42; border-radius: 999px; padding: 2px 10px; font-size: 12px;}
.se-warn {background: #FFF0E3; color: #8A3B12; border-radius: 999px; padding: 2px 10px; font-size: 12px;}
.se-times {background: #FFF0E3; color: #8A3B12; border-radius: 999px; padding: 2px 10px; font-size: 13px; font-weight: 600;}
.se-cheer {background: #E6F5EC; border-radius: 20px; padding: 18px 24px; display: flex; align-items: center; gap: 18px;}
.se-cheer p {margin: 0; font-size: 16px; font-weight: 600; color: #1F6B42;}
button[data-baseweb="tab"] p {font-size: 16px;}
/* Sidebar: push the privacy note and the API status to the bottom left. */
[data-testid="stSidebarContent"] {display: flex; flex-direction: column;}
[data-testid="stSidebarUserContent"] {flex: 1; display: flex; flex-direction: column; padding-bottom: 1.5rem;}
[data-testid="stSidebarUserContent"] > div {flex: 1; display: flex; flex-direction: column;}
.st-key-se_bottom, [data-testid="stLayoutWrapper"]:has(> .st-key-se_bottom) {margin-top: auto;}
</style>
""")

# --- Mascot and bowls (pure SVG, no image file needed) -------------------------
# Streamlit removes raw <svg> tags for security, so each drawing is embedded as an image.
def svg_img(svg: str, width: int, alt: str = "") -> str:
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{width}" alt="{alt}" style="flex-shrink:0">'


_BOX = ('<rect x="10" y="28" width="100" height="74" rx="18" fill="{body}"/>'
        '<rect x="10" y="20" width="100" height="22" rx="11" fill="{lid}"/>'
        '<rect x="50" y="10" width="20" height="14" rx="5" fill="{lid}"/>')
_FACES = {
    "happy": '<circle cx="44" cy="62" r="7" fill="#2B2118"/><circle cx="76" cy="62" r="7" fill="#2B2118"/>'
             '<circle cx="46" cy="60" r="2" fill="#FFFFFF"/><circle cx="78" cy="60" r="2" fill="#FFFFFF"/>'
             '<path d="M48 78 Q60 90 72 78" stroke="#2B2118" stroke-width="4" fill="none" stroke-linecap="round"/>'
             '<circle cx="32" cy="76" r="6" fill="#FFB199"/><circle cx="88" cy="76" r="6" fill="#FFB199"/>',
    "cheering": '<path d="M36 62 Q44 54 52 62" stroke="#2B2118" stroke-width="4" fill="none" stroke-linecap="round"/>'
                '<path d="M68 62 Q76 54 84 62" stroke="#2B2118" stroke-width="4" fill="none" stroke-linecap="round"/>'
                '<path d="M46 76 Q60 92 74 76 Z" fill="#2B2118"/>',
    "curious": '<circle cx="44" cy="62" r="8" fill="#2B2118"/><circle cx="76" cy="62" r="8" fill="#2B2118"/>'
               '<circle cx="47" cy="59" r="2.5" fill="#FFFFFF"/><circle cx="79" cy="59" r="2.5" fill="#FFFFFF"/>'
               '<ellipse cx="60" cy="82" rx="6" ry="5" fill="#2B2118"/>',
    "asleep": '<path d="M36 64 H52 M68 64 H84" stroke="#2B2118" stroke-width="4" stroke-linecap="round"/>'
              '<path d="M52 82 H68" stroke="#2B2118" stroke-width="4" stroke-linecap="round"/>',
}


def mascot(mood: str = "happy", width: int = 120) -> str:
    colours = {"body": "#F3D9C8", "lid": "#B9A090"} if mood == "asleep" else {"body": "#FF8A65", "lid": "#C2410C"}
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="110" viewBox="0 0 120 110">'
           f'{_BOX.format(**colours)}{_FACES[mood]}</svg>')
    return svg_img(svg, width, f"StudEat mascot, {mood}")


# Bowl colours from the mockups: (soup, topping 1, topping 2). Never alcohol or pork imagery.
_BOWLS = [("#F4A259", "#BC4B51", "#5B8E7D"), ("#FFF3D6", "#8AB17D", "#E76F51"), ("#E76F51", "#2A9D8F", "#E9C46A"),
          ("#F6E7CB", "#DD6E42", "#4F6D7A"), ("#C1440E", "#F2CC8F", "#81B29A"), ("#E9C46A", "#264653", "#2A9D8F")]


def bowl(meal_id: int, width: int = 64) -> str:
    soup, a, b = _BOWLS[meal_id % len(_BOWLS)]
    return svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48" viewBox="0 0 64 48">'
                   f'<ellipse cx="32" cy="20" rx="26" ry="10" fill="{soup}"/><circle cx="23" cy="17" r="5" fill="{a}"/>'
                   f'<circle cx="39" cy="16" r="4" fill="{b}"/><path d="M6 20 Q32 56 58 20 Z" fill="#E9D8C4"/></svg>',
                   width)


CHECK = svg_img('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path '
                'd="M5 12l5 5L19 7" stroke="#1F6B42" stroke-width="3" fill="none" stroke-linecap="round"/></svg>', 12)


# --- Words shown to the student --------------------------------------------------
def diet_label(d: str) -> str:
    return "No restriction" if d == "none" else d.capitalize()


def allergen_label(a: str) -> str:
    return "Tree nuts" if a == "nuts" else a.capitalize()


REACTIONS = {"Loved it": 5, "Good": 4, "So-so": 3, "Not for me": 1, "Didn't eat": None}


def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=10)


def fetch_box():
    r = api_get(f"/recommend/{st.session_state['user_id']}")
    st.session_state["box"] = (r.ok, r.json())
    st.session_state["ratings_sent"] = False


def new_box_after_rating():
    """Button callback: fetch the box again (now personalised) and start fresh reactions."""
    st.session_state["round"] = st.session_state.get("round", 0) + 1  # new widget keys = empty reactions
    fetch_box()
    st.session_state["new_box_ready"] = True


# --- Header: logo above the name -----------------------------------------------
st.image(LOGO, width=170)
st.html('<h1 class="se-title">Stud<span>Eat</span></h1>'
        '<p class="se-tagline">Eat well. Spend less. Cook never ;)</p>')

# --- Sidebar: the mascot talks -----------------------------------------------------
with st.sidebar:
    st.html('<div class="se-bubble"><b>Hey there!</b><br>Tell me what you like, I\'ll fill your week with good food.</div>'
            + mascot("happy", 110)
            + '<p style="margin:18px 0 8px;font-weight:600;font-size:18px">How it works</p>'
            '<ol style="margin:0;padding-left:20px;font-size:15px;line-height:1.9">'
            '<li>Tell us how you eat</li><li>Get your weekly box</li><li>Rate your meals</li></ol>'
            '<p class="se-muted" style="margin:10px 0 18px;font-size:14px">Every box gets a little better.</p>')
    bottom = st.container(key="se_bottom")  # stays at the bottom left of the sidebar
    bottom.html('<div class="se-note">Every way of eating is welcome here. Your choices stay private and are only '
                'used to build your box.</div>')

try:
    health = api_get("/health").json()
    options = api_get("/options").json()
except requests.exceptions.RequestException:
    st.html(f'<div class="se-panel se-header">{mascot("asleep", 90)}<div><h2>The kitchen is closed</h2>'
            f'<p>The API is not reachable at {html.escape(API_URL)}. Start it with: '
            '<code>python -m uvicorn api.main:app --reload</code></p></div></div>')
    st.stop()

with bottom:
    if "user_id" in st.session_state:
        st.caption(f"Signed in as **{st.session_state['name']}** (user {st.session_state['user_id']})")
    st.caption(f"API connected · {health['meals_in_catalogue']} meals · catalogue: {health.get('catalogue', 'mock')}")

tab_signup, tab_box, tab_rate = st.tabs(["1. Sign up", "2. My weekly box", "3. Rate my meals"])

# --- Page 1: sign up ---------------------------------------------------------------
with tab_signup:
    with st.container(border=True):
        name = st.text_input("What should we call you?", max_chars=40)
        diet = st.pills(
            "How do you eat? All options are equal, listed A to Z.",
            sorted(options["diets"], key=diet_label), format_func=diet_label, key="diet",
        )
        allergies = st.pills(
            "Anything you can't eat? It will never be in your box.",
            options["allergens"], selection_mode="multi", format_func=allergen_label, key="allergies",
            help="Peanuts are not tree nuts: if you are allergic to both, tick both.",
        )
        col_budget, col_meals = st.columns([2, 1], gap="large")
        with col_budget:
            budget = st.slider("Your weekly budget", 20, 80, (35, 50), format="%d €")
            st.caption(f"{budget[0]} € to {budget[1]} € a week · about "
                       f"{round(budget[0] * 52 / 12)} € to {round(budget[1] * 52 / 12)} € a month")
        with col_meals:
            meals = st.number_input("Meals per week", min_value=1, max_value=14, value=10)
        build = st.button("Build my box", type="primary")

    if build:
        if not name.strip():
            st.warning("Tell us your first name first.")
        elif diet is None:
            st.warning("Pick how you eat (choose « No restriction » if nothing applies).")
        else:
            payload = {"name": name.strip(), "diet": diet, "allergies": allergies,
                       "budget_min": budget[0], "budget_max": budget[1], "meals_per_week": int(meals)}
            r = requests.post(f"{API_URL}/profile", json=payload, timeout=10)
            if r.ok:
                st.session_state.update(user_id=r.json()["user_id"], name=name.strip(), profile=payload)
                fetch_box()
                st.success(f"Your box is ready, {name.strip()}! Open « 2. My weekly box » above.")
            else:
                st.error(r.json().get("detail", "Could not create the profile."))

# --- Page 2: my weekly box ---------------------------------------------------------
with tab_box:
    if "user_id" not in st.session_state:
        st.info("Create your profile first, in « 1. Sign up ».")
    else:
        p = st.session_state["profile"]
        details = [diet_label(p["diet"])] + [f"no {allergen_label(a).lower()}" for a in p["allergies"]]
        details.append(f"{p['meals_per_week']} meals")
        col_head, col_btn = st.columns([5, 1.3], vertical_alignment="center")
        with col_head:
            st.html(f'<div class="se-header">{mascot("cheering", 72)}<div>'
                    f'<h2>Your box for this week, {html.escape(st.session_state["name"])}</h2>'
                    f'<p>{" · ".join(details)}</p></div></div>')
        with col_btn:
            if st.button("Refresh my box", icon=":material/refresh:"):
                fetch_box()

        ok, data = st.session_state["box"]
        if not ok:
            st.error(data.get("detail", "Could not get recommendations."))
        else:
            total, budget_max = data["total_price"], p["budget_max"]
            per_meal = total / len(data["meals"]) if data["meals"] else 0
            st.html(f'<div class="se-panel"><div class="se-row"><span><b>{total:.2f} €</b> of your '
                    f'{budget_max} € budget</span><span class="se-muted">about {per_meal:.2f} € a meal</span></div>'
                    f'<div class="se-bar"><div style="width:{min(100, 100 * total / budget_max):.0f}%"></div></div></div>')
            if not data["within_budget"]:
                st.info("This box costs less than your minimum budget. Good news for your wallet!")

            # A meal can appear twice in the box: show it once with « × 2 ».
            counts = Counter(m["id"] for m in data["meals"])
            unique = list({m["id"]: m for m in data["meals"]}.values())
            cards = []
            for m in unique:
                chips = ("".join(f'<span class="se-warn">Contains: {allergen_label(a).lower()}</span>'
                                 for a in m["allergens"])
                         or f'<span class="se-ok">{CHECK}No listed allergen</span>')
                kcal = f'{m["calories"]:.0f} kcal' if m.get("calories") else ""
                times = f'<span class="se-times">× {counts[m["id"]]}</span>' if counts[m["id"]] > 1 else ""
                cards.append(f'<article class="se-card"><div class="se-row">{bowl(m["id"])}{times}</div>'
                             f'<h3>{html.escape(m["name"])}</h3>'
                             f'<div class="se-row"><span class="se-muted" style="font-size:13px">{kcal}</span>'
                             f'<span class="se-price">{m["price"]:.2f} €</span></div>'
                             f'<div class="se-chips">{chips}</div></article>')
            st.html(f'<div class="se-grid">{"".join(cards)}</div>')

            source = data.get("model", "")
            if source == "model":
                st.caption("Personalised from your ratings.")
            elif source == "popularity":
                st.caption("Student favourites for now. Rate your meals and your next box will be made for you.")
            else:
                st.caption(f"Demo mode ({source}): the recommendation model is not loaded.")
            st.caption("Allergens are listed with words, not only colours. Check each label if you have a severe allergy.")

# --- Page 3: rate my meals ---------------------------------------------------------
with tab_rate:
    box = st.session_state.get("box")
    if not box or not box[0]:
        st.info("Get your weekly box first, then come back to tell us how it was.")
    else:
        meals = {m["id"]: m["name"] for m in box[1]["meals"]}  # one reaction per distinct meal
        st.html(f'<div class="se-header">{mascot("curious", 80)}<div><h2>So, how was your week?</h2>'
                "<p>One tap per meal. Skip the ones you didn't eat. Your next box learns from every answer.</p>"
                '</div></div>')

        rnd = st.session_state.get("round", 0)
        answered = [mid for mid in meals if st.session_state.get(f"react_{rnd}_{mid}")]
        st.html(f'<div class="se-panel se-row" style="gap:20px"><b style="white-space:nowrap">'
                f'{len(answered)} of {len(meals)} answered</b><div class="se-bar" style="flex:1;margin:0">'
                f'<div style="width:{100 * len(answered) / len(meals):.0f}%"></div></div>'
                '<span class="se-muted" style="white-space:nowrap">Every answer makes your next box smarter</span></div>')
        # A meal you answered gets an orange border, like in the mockup.
        if answered:
            st.html("<style>" + "".join(f".st-key-row_{mid} {{border: 2px solid #C2410C !important;}}"
                                        for mid in answered) + "</style>")

        for meal_id, meal_name in meals.items():
            with st.container(border=True, key=f"row_{meal_id}"):
                c_bowl, c_name, c_react = st.columns([0.7, 3, 5], vertical_alignment="center")
                c_bowl.html(bowl(meal_id, 56))
                c_name.markdown(f"**{meal_name}**")
                c_react.segmented_control("Your reaction", list(REACTIONS), key=f"react_{rnd}_{meal_id}",
                                          label_visibility="collapsed")

        sent = st.button("Send my ratings", type="primary", disabled=st.session_state.get("ratings_sent", False))
        st.caption("Loved it = 5 · Good = 4 · So-so = 3 · Not for me = 1. « Didn't eat » sends nothing.")

        if sent:
            given = {mid: REACTIONS[st.session_state[f"react_{rnd}_{mid}"]] for mid in answered}
            given = {mid: score for mid, score in given.items() if score is not None}
            if not given:
                st.warning("Tap a reaction on at least one meal you ate.")
            else:
                errors = 0
                for meal_id, rating in given.items():
                    r = requests.post(f"{API_URL}/log_interaction", timeout=10, json={
                        "user_id": st.session_state["user_id"], "meal_id": meal_id, "rating": rating})
                    errors += not r.ok
                if errors:
                    st.error(f"{errors} rating(s) could not be saved. Try again.")
                else:
                    st.session_state["ratings_sent"] = True
                    st.balloons()

        if st.session_state.get("ratings_sent"):
            st.html(f'<div class="se-cheer">{mascot("cheering", 56)}<p>Got it! Your next box just got smarter.</p></div>')
            st.button("See my new box", icon=":material/arrow_forward:", on_click=new_box_after_rating)
        if st.session_state.pop("new_box_ready", False):
            st.success("Your new box is ready! Open « 2. My weekly box » above.")
