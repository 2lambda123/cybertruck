
# CyberTruck - Distracted Driver Detection

CAP6411 Fall 2023 - Group 6   
Ron, Robin, Suneet, Osi, Kasun

## Objective

* Driver distraction is one of the leading causes of fatal accidents.
* The implementation of a warning tool for drivers during these situations can potentially save lives.
* Project objective: Develop a warning tool to assist in regaining driver attention.

# Repo setup

- `detection` - Contains all the python files for the detection model
  - `face_detection` code pertaining to detecting facial features.
  - `hands_detection` code pertaining to detecting hand features.
  - `model.py`
- `client` - Contains all of the code for the android app.

If you want to add a new module create a folder with an `__init__.py` file and import code into it that you want to expose.
