import cohere
from cohere import ClassifyExample
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# print(f"COHERE_API_KEY:{COHERE_API_KEY}")
co = cohere.Client(
  api_key=COHERE_API_KEY, # This is your trial API key
) 

response = co.classify(
  model='embed-english-v3.0',
  inputs=["Good evening"],
  examples=[ClassifyExample(text="Good morning", label="positive"),
            ClassifyExample(text="Good night", label="positive"),
            ClassifyExample(text="Stop that", label="negativ"),
            ClassifyExample(text="Do not come here", label="negativ")])
print('The confidence levels of the labels are: {}'.format(response.classifications))

"""The confidence levels of the labels are: [ClassifyResponseClassificationsItem(id='62850c4b-517c-4e24-a066-ed57693481bc', input='good evening', prediction='positive', predictions=['positive'], confidence=0.5811586, confidences=[0.5811586], labels={'negativ': ClassifyResponseClassificationsItemLabelsValue(confidence=0.41884142), 'positive': ClassifyResponseClassificationsItemLabelsValue(confidence=0.5811586)}, classification_type='single-label')]
"""