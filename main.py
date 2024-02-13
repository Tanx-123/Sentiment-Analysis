from flask import Flask,jsonify,Response, render_template
from flask_cors import CORS
import tensorflow as tf 
import transformers
from transformers import AutoTokenizer
from flask_restful import Resource, Api, reqparse

new_model = tf.keras.models.load_model('sentiment_model',custom_objects={"TFBertModel": transformers.TFBertModel})
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

app = Flask(__name__)
api = Api(app)
CORS(app)

class index(Resource):    
     def get(self):
         return Response(render_template('index.html'))

class SentimentAnalysis(Resource):
    def post(self):
        # Get the input data from the request
        parser = reqparse.RequestParser()
        parser.add_argument('text', type=str, help='Text to analyze', required=True)
        args = parser.parse_args()
        text = args['text']
        token = tokenizer.encode_plus(text, 
                                add_special_tokens=True,
                                max_length=50, 
                                truncation=True,
                                padding="max_length",
                                return_token_type_ids=False,
                                return_attention_mask=True,
                                return_tensors='tf')
        input_ids = token['input_ids']
        attention_mask = token['attention_mask']
        # Make a prediction
        sentiment = new_model.predict([input_ids,attention_mask])[0]
        sentiment_labels = {0:'You seemed to be angry.', 
                        1:'Are you afraid?', 
                        2:'Good to see you happy.',
                        3:'Surprise!!!'}
        sentiment = sentiment_labels[sentiment.argmax()]
        # Return the prediction as JSON
        return jsonify({'sentiment': sentiment})
    
api.add_resource(index,'/')
api.add_resource(SentimentAnalysis, '/predict')

# Start the API server
if __name__ == '__main__':
    app.run(debug=True)