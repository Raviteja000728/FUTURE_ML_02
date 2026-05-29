import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Support Ticket Classification",
    page_icon="🎫",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:bold;
    color:#00C8FF;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------

st.markdown(
    "<p class='main-title'>🎫 Support Ticket Classification System</p>",
    unsafe_allow_html=True
)

st.write("Automatically classify customer support tickets and assign priority.")

# -----------------------------
# TRAINING DATA
# -----------------------------

data = {
    "text":[
        "internet not working",
        "wifi connection issue",
        "payment failed",
        "refund not received",
        "cannot login account",
        "forgot password",
        "need product information",
        "feature request for app",
        "server down urgently",
        "billing amount incorrect",
        "account locked",
        "application crashing"
    ],

    "category":[
        "Technical",
        "Technical",
        "Billing",
        "Billing",
        "Account",
        "Account",
        "General",
        "Feature Request",
        "Technical",
        "Billing",
        "Account",
        "Technical"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# NLP MODEL
# -----------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()

model.fit(X, df["category"])

# -----------------------------
# USER INPUT
# -----------------------------

st.sidebar.header("Ticket Details")

customer_name = st.sidebar.text_input(
    "Customer Name",
    "Raviteja"
)

ticket_title = st.sidebar.text_input(
    "Ticket Title",
    "Internet Issue"
)

ticket_description = st.sidebar.text_area(
    "Ticket Description",
    "My internet is not working and I need urgent help."
)

department = st.sidebar.selectbox(
    "Department",
    [
        "Technical Support",
        "Billing Support",
        "Account Support",
        "General Support"
    ]
)

# -----------------------------
# CLASSIFICATION
# -----------------------------

full_text = ticket_title + " " + ticket_description

input_vector = vectorizer.transform(
    [full_text]
)

prediction = model.predict(
    input_vector
)[0]

# -----------------------------
# PRIORITY LOGIC
# -----------------------------

high_keywords = [
    "urgent",
    "immediately",
    "critical",
    "down",
    "not working",
    "crashing"
]

medium_keywords = [
    "issue",
    "problem",
    "unable",
    "error"
]

text = full_text.lower()

priority = "Low 🟢"

for word in high_keywords:
    if word in text:
        priority = "High 🔴"

for word in medium_keywords:
    if word in text and priority != "High 🔴":
        priority = "Medium 🟡"

# -----------------------------
# TEAM ASSIGNMENT
# -----------------------------

team_map = {
    "Technical":"Technical Team",
    "Billing":"Billing Team",
    "Account":"Account Team",
    "General":"Customer Service",
    "Feature Request":"Product Team"
}

assigned_team = team_map.get(
    prediction,
    "Support Team"
)

# -----------------------------
# SLA
# -----------------------------

if "High" in priority:
    sla = "1 Hour"

elif "Medium" in priority:
    sla = "4 Hours"

else:
    sla = "24 Hours"

# -----------------------------
# RESULTS
# -----------------------------

st.subheader("📊 Ticket Analysis")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Category",
    prediction
)

col2.metric(
    "Priority",
    priority
)

col3.metric(
    "Response SLA",
    sla
)

st.divider()

# -----------------------------
# TICKET SUMMARY
# -----------------------------

st.subheader("📝 Ticket Summary")

summary = pd.DataFrame({
    "Field":[
        "Customer",
        "Department",
        "Category",
        "Priority",
        "Assigned Team"
    ],

    "Value":[
        customer_name,
        department,
        prediction,
        priority,
        assigned_team
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------

st.subheader("💡 AI Recommendations")

if "High" in priority:

    st.error(
        f"""
        Critical issue detected.

        Assign immediately to:
        {assigned_team}

        Target response:
        {sla}
        """
    )

elif "Medium" in priority:

    st.warning(
        f"""
        Moderate priority issue.

        Assign to:
        {assigned_team}

        Response target:
        {sla}
        """
    )

else:

    st.success(
        f"""
        Standard support request.

        Assign to:
        {assigned_team}

        Response target:
        {sla}
        """
    )

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.caption(
    "Machine Learning Internship Project | Support Ticket Classification System"
)