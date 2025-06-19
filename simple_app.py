import streamlit as st

st.title("Simple Test App")
st.write("This is a basic Streamlit app to test the environment.")

# Test basic functionality
if st.button("Click me"):
    st.success("Button clicked successfully!")

st.write("Environment test complete.")