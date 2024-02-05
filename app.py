from flask import Flask, request, app, render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

## Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

## Route for prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Schizophrenia=request.form.get('Schizophrenia'),
            Depressive=request.form.get('Depressive'),
            Anxiety=request.form.get('Anxiety'),
            Bipolar=request.form.get('Bipolar'),
            Eating=request.form.get('Eating')
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        results = ['A' if results[0] == 0 else 'B']
        return render_template('index.html',results=results[0])


if __name__=="__main__":
    app.run()