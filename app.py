from flask import Flask, render_template, request
from wtforms import Form, validators, StringField
from pathlib import Path
from joblib import load

pred_model_path = Path('./models')
pred_model_filename = 'pipeline_naivebayes.joblib'
pred_model = load(pred_model_path / pred_model_filename)
app = Flask(__name__)

# Model
class InputForm(Form):
    r = StringField(validators=[validators.InputRequired()])

# View
@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        r = form.r.data
        tokens = [r]
        y = pred_model.predict(tokens)
        if y == 1:
            s = 'Positive'
        elif y == 0:
            s = 'Negative'
    else:
        s = None
    return render_template("view.html", form=form, s=s)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)