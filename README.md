# Dynamic Entity Recognizer

An interactive, asynchronous entity recognizer built with Gradio and gemini-2.5-flash.


This project uses the in-context learning capabilities of Gemini, allowing users to define entities, provide training images, and get immediate classification results without any model fine-tuning or prior machine learning knowledge.


## How it works:
### 1.  Dynamic Prompt Construction:
For each test image, the application sends the training images and their corresponding entity names (the "few shots") in a single prompt to the model.

### 2. In-Context Learning: 
The model uses the examples provided to understand the characteristics of each entity. This is temporary and lasts only for the duration of that single API call.

### 3. Inference:
The model is then asked to identify any of the learned entities within the new test image.

### 4. Structured Output:
The model's response is formatted as a JSON object, which is then parsed by a dynamically generated Pydantic model to ensure further data integrity.

## Installation: 

### 1. Clone the repository:
  ``` bash
   git clone https://github.com/mosheragomaa/dynamic-entity-recognizer.git
   cd dynamic-entity-recognizer
  ```


### 2. Install the required dependencies:
```bash
  pip install -r requirements.txt
```
 

### 3. Add your API key:

Create a file named .env in the root of the project directory.

Add your API key to it: 
```bash 
API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

## Usage
### 1. Run the UI by executing the app.py script from your terminal: `python app.py`

### 2. The terminal will provide a local URL (e.g., http://127.0.0.1:7860). Open this link in your web browser.

### 3. Configure the classifier:
  - Select Number of Entities: Choose how many distinct categories you want to classify.
  - Define Entities: Enter a name for each entity.
  - Upload Training Images: For each entity, upload a folder containing its sample images.
  - Upload Test Images: Upload a folder of new images you want the model to classify.
  - Submit.

The results will in sorted galleries labeled by entity name.
