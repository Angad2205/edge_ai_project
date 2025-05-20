# Steps for Model 
Sign_detection
First Upload the best.pt file in the note book from the github repo and then follow the steps 

Note -> i have trained the model in collab so if by chance any error comes so just do gpt it will convet the command compaitable to juipter note bokk althought if have done it alread but still if their is the problem then you can do that 

# Step 1: Clone YOLOv5 Repository
 
!git clone https://github.com/ultralytics/yolov5

# Step 2: Change Directory to yolov5

import os
os.chdir('yolov5')

# Step 3: Install Dependencies

!pip install -r requirements.txt

# If using the virtual env then also run this 

!pip install seaborn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 4: Upload Your Trained Weights (best.pt)

from google.colab import files  # Ignore if using local Jupyter

uploaded = files.upload()       # This will prompt file upload

# Step 5: Load the Model
 
 import cv2
from matplotlib import pyplot as plt

# Load your test image
img = cv2.imread('your_image.jpg')  # replace with your image name

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img)

# Show result
results.print()

plt.imshow(np.squeeze(results.render()))

plt.axis('off')

plt.show()

# Use camera function of the Jetbot 

![Pred](https://github.com/user-attachments/assets/c7d13fad-822f-439d-843a-53c9b0ba47c9)


