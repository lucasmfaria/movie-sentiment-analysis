# movie-sentiment-analysis
Study of sentiment analysis techniques using Stanford "Large Movie Review Dataset".

The dataset can be downloaded in: http://ai.stanford.edu/~amaas/data/sentiment/

The project steps reproduced here are:
> Study: jupyter notebooks for the data exploration, model construction, model selection and model evaluation
>
> Training/prototyping: script creation for training prodution model
> 
> Deploying: use of Flask to serve the model to the user

# Docker
You can build a docker image and run the Flask server within a container with these commands:
- docker build . -t <your_image_tag> (create image)
- docker run <your_image_id> -d -p 5000:5000 (create container)

# Local/virtual env
You can run the code within your local or virtual python environment with these commands:
- pip install -r requirements.txt
- python app.py
