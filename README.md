# Cherry Leaf Powdery Mildew Detector

Our Cherry Leaf Mildew Detector is a full end-to-end predictive analytics solution built to automate the detection of powdery mildew on cherry leaves using a dataset of over 4,000 labeled leaf images.

This README lays out the project's pathway through the CRISP-DM process:

1. **Data Collection & Preparation**: Download and clean the Kaggle cherry-leaves dataset, then split it into train/validation/test with a fixed seed.  
2. **Data Visualization**: Explore class balance, image dimensions, per-class mean/std images, variance maps, and sample montages.  
3. **Modeling & Evaluation**: Define, train, and tune a custom CNN with EarlyStopping; export metrics, history, and artifacts.  
4. **Interactive Dashboard**: Deploy a multipage Streamlit app that documents objectives, visual insights, preprocessing controls, training curves, validation metrics, and a live “Diagnosis Station” for user uploads.

Scroll down to learn more about the dataset, business requirements, hypotheses, dashboard design, and deployment instructions.

## Deployment

### Render

- The App live link is: `https://pp5-mildewdetection.onrender.com/`
- Set the .python-version file Python version to 3.12.
- The project was deployed to Render using the following steps.

1. Ensure models/*.h5 is included in your .gitignore file as it contains too much data to submit to GitHub.
2. Go to render.com and create a new Web Service. **Add New** then **Web Service**.
3. Connect your GitHub account and select your repository (and branch if you have multiple).
4. We chose the **Standard** payment model which permits 2 GB RAM as our project does not fit in the smaller tiers.
5. The build command is left as default (pip install -r requirements.txt ).
6. The start command is set to run a Streamlit server with an exported port and a server flag that permits external traffic:
   - streamlit run app.py --server.port $PORT --server.address 0.0.0.0
7. No environment variables are required.
8. Select **Manual Deploy** then **Deploy Latest Commit**.
9. Once "Live" turns green, the URL provided will allow you to access the deployed project.
