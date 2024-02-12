from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import pickle

import tensorflow as tf
def create_my_app():
    app = Flask(__name__)

    # Load preprocessed pipeline
    preprocessing_pipe = pickle.load(open('pipepreprocessing.pkl', 'rb'))
    preprocessing_pipe_uval=pickle.load(open('pipepreprocessingu_val.pkl', 'rb'))
    # Load Keras model
    keras_model = load_model('my_keras_model.keras')
    keras_model_uval = load_model('my_uval_model.keras')


    @app.route("/")
    def hello_world():
        return render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def predict_savings():
        # Get form data
        climaticZone = request.form.get('ClimaticZone')
        shadeDirection = request.form.get('ShadeDirection')
        buildingOrientation = int(request.form.get('BuildingOrientation'))
        latitude = float(request.form.get('Latitude'))
        longitude = float(request.form.get('Longitude'))
        shadeExtent = float(request.form.get('ShadeExtent'))
        roofUValue = float(request.form.get('RoofUValue'))
        heightOfShade = int(request.form.get('HeightofShade'))
        shadingTransmittance = float(request.form.get('ShadingTransmittance'))
        
        def combined_nearest_values(value1, value2):
            """Find the combined nearest values from two arrays."""
            arr1 = np.array([17.67, 19.88, 23.26, 26.85, 28.61, 31.64, 30.32, 18.52, 21.17,
                    23.03, 24.58, 26.92, 28.02, 15.14, 17.38, 21.15, 12.97,  8.51,
                    10.79, 12.92, 13.08, 17.69, 15.49, 19.07, 22.57, 25.57])
            arr2 = np.array([75.9064, 75.3433, 77.4126, 80.9462, 77.209 , 74.8737, 78.0322,
                    73.8567, 72.8311, 72.5714, 73.7125, 75.7873, 73.3119, 76.9214,
                    78.4867, 79.0882, 77.5946, 76.9366, 78.7047, 74.856 , 80.2707,
                    83.2185, 73.8278, 72.8777, 88.3639, 91.8933])
            
            min_diff = float('inf')
            nearest_combined_values = None
            
            for v1 in arr1:
                for v2 in arr2:
                    diff = abs(v1 - value1) + abs(v2 - value2)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_combined_values = (v1, v2)
            
            return nearest_combined_values
        latitude,longitude = combined_nearest_values(latitude,longitude)
    

        # Preprocess input data
        input_data = np.array([climaticZone, shadeDirection,  buildingOrientation, latitude, longitude, shadeExtent, roofUValue, heightOfShade,
                            shadingTransmittance]).reshape(1, -1)
        preprocessed_input = preprocessing_pipe.transform(input_data)
        preprocessed_input = preprocessed_input.astype(np.float64)
        

        # Make predictions
        result = keras_model.predict(preprocessed_input)
        input_data_for_uval=np.array([latitude, longitude,result[0][0]]).reshape(1,-1)
        preprocessed_input_uval = preprocessing_pipe_uval.transform(input_data_for_uval)
        preprocessed_input_uval = preprocessed_input_uval.astype(np.float64)
        result_uval = keras_model_uval.predict(preprocessed_input_uval)
  
        return render_template('predict.html', result=result[0][0], result_uval=result_uval[0][0])
    return app
    
