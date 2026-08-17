
import streamlit as st
from Auth.Auth import signup_user, login_user

def show_auth_page():

    st.title("🔐 Startup Success Predictor")
    st.subheader("Login or Create an Account")

    login_tab, signup_tab = st.tabs(
        ["Login", "Sign Up"]
    )

    # LOGIN TAB
    with login_tab:
        username = st.text_input("Username",key="login_username")
        password = st.text_input("Password",type="password",key="login_password")
        if st.button("Login"): #use_container_width=True
            if not username or not password:
                st.warning(
                    "Please enter username and password."
                )
            elif login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error(
                    "Invalid username or password."
                )

    # SIGNUP TAB
    with signup_tab:
        new_username = st.text_input("Choose Username",key="signup_username")
        new_password = st.text_input("Choose Password",type="password",key="signup_password")
        confirm_password = st.text_input("Confirm Password",type="password",key="confirm_password")

        if st.button(
            "Create Account",
            # use_container_width=True
        ):

            if not new_username or not new_password:
                st.warning("Please fill all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if signup_user(new_username, new_password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists.")
