Lane Line Detection using OpenCV

About the Project

This project demonstrates road lane line detection using
Python and OpenCV. It processes road video frames and
identifies candidate lane lines using classical computer
vision techniques.

Technologies Used

- Python
- OpenCV
- NumPy
- MoviePy
- Tkinter

Features

- Video processing
- Grayscale image conversion
- Canny edge detection
- Region of Interest (ROI) masking
- Hough Line Transform
- Lane line visualisation

Project Structure

- `main.py` - Main lane detection implementation
- `gui.py` - Graphical user interface
- `requirements.txt` - Required Python packages

Installation

Clone this repository and navigate to its directory.

Install the required dependencies:

    pip install -r requirements.txt

Run the Project

Run the main program:

    python main.py

To launch the graphical interface:

    python gui.py

Limitations

Detection accuracy depends on road visibility, lighting,
camera position, and the selected image-processing parameters.
This is a classical computer vision implementation, not a
trained machine learning model.
