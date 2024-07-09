import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Get current date
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Auto-refresh every 2 seconds
st_autorefresh(interval=2000, limit=None, key="fizzbuzzcounter")

# File path for the attendance file
attendance_file_path = f"attendence system\\Attendance\\Attendance_{date}.csv"

# Check if the attendance file exists and read it
try:
    df = pd.read_csv(attendance_file_path)
    st.dataframe(df.style.highlight_max(axis=0))
except FileNotFoundError:
    st.write(f"No attendance data found for {date}.")

# Optional: Display current date and time for user reference
st.write(f"Current Date: {date}")
st.write(f"Current Time: {timestamp}")
