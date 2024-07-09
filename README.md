# Attendance System Using Face Recognition

This project is a face recognition-based attendance system using Python. It leverages OpenCV for face detection, scikit-learn for machine learning, and Streamlit for a web-based interface.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/attendance-system.git
    cd attendance-system
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the `haarcascade_frontalface_default.xml` file and place it in the `attendence system\data` directory:
    ```bash
    wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml -P attendence system/data
    ```

## Usage

1. **Collecting Data**:
    Run the `collecting_data.py` script to collect face data:
    ```bash
    python collecting_data.py
    ```

2. **Recognition of Image**:
    Run the `recognition_of_image.py` script to recognize faces:
    ```bash
    python recognition_of_image.py
    ```

3. **Streamlit Application**:
    Run the Streamlit app to display attendance data:
    ```bash
    streamlit run app.py
    ```

4. **Testing**:
    Run the `test.py` script to test the recognition and attendance logging:
    ```bash
    python test.py
    ```

## Project Structure

attendance-system/
│
├── attendence system/
│ ├── data/
│ │ ├── faces_data.pkl
│ │ ├── haarcascade_frontalface_default.xml
│ │ └── names.pkl
│ └── Attendance/
│ └── Attendance_<date>.csv
│
├── collecting_data.py
├── recognition_of_image.py
├── app.py
├── test.py
├── requirements.txt
└── README.md

## Notes

- If you are using this code, make sure to change the paths of the files according to your environment. This includes paths in the scripts like `collecting_data.py`, `recognition_of_image.py`, `app.py`, and `test.py`. Adjust the paths to match the directory structure on your local machine.
- Ensure the `haarcascade_frontalface_default.xml` file is correctly downloaded and placed in the `attendence system\data` directory.
