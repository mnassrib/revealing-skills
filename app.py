from flask import Flask, render_template, url_for, request
import pandas as pd
import spacy

# loading the model
nlp = spacy.load("revealing_skills_model")

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/revealing', methods=["POST"])
def revealing():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		doc = nlp(rawtext)
		d = []
		if len(doc.ents)==0:
			return render_template("index.html", rawtext=rawtext, message="Pas de compétences révélées dans cet exemple !")
		else:
			for ent in doc.ents:
				d.append((ent.label_, ent.text))
				df = pd.DataFrame(d, columns=('named entity', 'output'))
				SKILLS_named_entity = df.loc[df['named entity'] == 'Compétences']['output']
				results = SKILLS_named_entity
				num_of_results = len(results)           
			return render_template("index.html", rawtext=rawtext, results=results, num_of_results=num_of_results)

if __name__ == '__main__':
	app.run(debug=True)