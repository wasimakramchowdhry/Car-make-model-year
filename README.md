# Car-make-model-year

This project sets up a CCTV detection system using computer vision and machine learning. Below is a step-by-step guide to set up the project, including necessary Python files, directory structure, and installation instructions.

# Step-by-Step Setup
  ### 1. Create Project Directory and Navigate into It
```shell
mkdir cctv-detection-app
cd cctv-detection-app
```
### 2. Set Up Virtual Environment
```shell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3. Create Project Structure
```shell
mkdir data data\raw data\processed data\models
mkdir notebooks
mkdir src src\capture src\detection src\analysis src\models src\utils
```
##### Create __init__.py files
```shell
New-Item -Path src\capture\__init__.py -ItemType File
New-Item -Path src\detection\__init__.py -ItemType File
New-Item -Path src\analysis\__init__.py -ItemType File
New-Item -Path src\models\__init__.py -ItemType File
New-Item -Path src\utils\__init__.py -ItemType File
```
##### Create main files
```shell
New-Item -Path src\app.py -ItemType File
New-Item -Path src\config.py -ItemType File
New-Item -Path Dockerfile -ItemType File
New-Item -Path requirements.txt -ItemType File
```
### 4. Install Required Packages
```shell
pip install opencv-python yolov5 deepface tensorflow torch jupyter
```
### 5. Populate requirements.txt
```shell
opencv-python
yolov5
deepface
tensorflow
torch
jupyter
```
### 6. Set Up Jupyter Notebook
```shell
jupyter notebook --generate-config
```
### 7. Launch Jupyter Notebook
``` shell
jupyter notebook
```
### 8. Running the Application
To run your application, execute individual scripts or run Jupyter notebooks for specific tasks.

For Docker:
### Build the Docker image
```bash
docker build -t cctv-detection-app .
```
### Run the Docker container
```bash
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 8888:8888 cctv-detection-app
```

### Create the environment in Anaconda:
```bash
conda create -n ml python=3.7.13
conda activate ml
```
#### Install the requirements:
```bash
pip install -r requirements.txt
```
## Running Python Files
##### Running the main application
```bash python src/app.py```

##### Set PYTHONPATH if needed and run the app
```bash
$env:PYTHONPATH = ".;src"
python src/app.py
```

##### For car detection and training
```bash
python src/models/saudi_plate_train_model.py
python src/detection/detect_plates.py
python run_app.py
```
##### For face recognition and analysis
```bash
python src/analysis/feature_extraction.py
python src/models/train_classifier.py
python run_app.py
```

### Public Repositories for Datasets

- [Roboflow - Car Class](https://universe.roboflow.com/project-vewd3/carclass)
- [Roboflow - Car Recognition](https://universe.roboflow.com/carr-5b5fq/carrecognition/browse?queryText=&pageSize=50&startingIndex=100&browseQuery=true)
- [Roboflow - Car Brand Classification](https://universe.roboflow.com/anpr-yyewx/car-brand-classification-guwpf)
- [Roboflow - Traffic Cars](https://universe.roboflow.com/traffic-ojgzy/cars-xjylt)

- [Hugging Face](https://huggingface.co/)
- [PyTorch Hub](https://pytorch.org/hub/)
- [TensorFlow Hub](https://tfhub.dev/)





## Author
**Wasim Akram Chowdhry**

- GitHub: [Wasim Akram Chowdhry](https://github.com/wasimakramchowdhry)
- LinkedIn: [Wasim Akram Chowdhry](https://www.linkedin.com/in/wasim-akram-chowdhry)
- Email: [waseemakramchaudhari@gmail.com](mailto:waseemakramchaudhari@gmail.com)

Experienced in chatbot development, conversational AI, and computer vision solutions, with a focus on platforms like IBM Watson, Google Dialogflow, and cutting-edge AI technologies. Always passionate about learning and sharing insights in the AI and machine learning space.

