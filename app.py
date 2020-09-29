from flask import Flask, render_template, url_for, request
import pandas as pd
import spacy

# loading the model
nlp = spacy.load("reveal_skills_model")

import fr_core_news_md
nlpfr = fr_core_news_md.load()

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
				if nlpfr(ent.text)[0].pos_ == "VERB":
					entmod = " ".join([str(nlpfr(ent.text)[0].lemma_)]+[str(token) for token in nlpfr(ent.text)[1:]])
					d.append((ent.label_, entmod.capitalize()))
				else:
					d.append((ent.label_, ent.text.capitalize()))
				df = pd.DataFrame(d, columns=('named entity', 'output'))
				SKILLS_named_entity = df.loc[df['named entity'] == 'Compétences']['output']
				results = SKILLS_named_entity
				num_of_results = len(results)  
			return render_template("index.html", rawtext=rawtext, results=results, num_of_results=num_of_results)

if __name__ == '__main__':
	app.run(debug=True)