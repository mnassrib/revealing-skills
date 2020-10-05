from flask import Flask, render_template, url_for, request
import pandas as pd
import spacy

# loading the model
nlp = spacy.load("revealing_skills_model")

#import fr_core_news_sm
#nlpfr = fr_core_news_sm.load()

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html", skills_entities={})

@app.route('/revealing', methods=["POST"])
def revealing():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		doc = nlp(rawtext)
		d = []
		if len(doc.ents)==0:
			return render_template("index.html", rawtext=rawtext, message="Pas de compétences révélées dans ce texte !")
		else:
			for ent in doc.ents:
				if nlp(ent.text)[0].pos_ == "VERB":
					entmod = " ".join([str(nlp(ent.text)[0].lemma_)]+[str(token) for token in nlp(ent.text)[1:]])
					d.append((ent.label_, entmod.capitalize()))
				else:
					d.append((ent.label_, ent.text.capitalize()))
				skills_entities = get_skills_entities(pd.DataFrame(d, columns=('named entity', 'output'))) 
			return render_template("index.html", rawtext=rawtext, skills_entities=skills_entities)

def get_skills_entities(df):
	skills_entity = {}
	for entit in pd.unique(df["named entity"]):
		skills_entity[entit] = df[df['named entity'] == entit]['output'].tolist()
	return skills_entity
	
if __name__ == '__main__':
	app.run(debug=True)