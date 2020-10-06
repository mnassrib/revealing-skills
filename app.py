from flask import Flask, render_template, url_for, request
import pandas as pd
import spacy
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the model
nlp = spacy.load("revealing_skills_model")

#import fr_core_news_sm
#nlpfr = fr_core_news_sm.load()

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html", skills_entities={}, resumeMatchPercentageJobs={})

@app.route('/revealing', methods=["POST"])
def revealing():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		doc = nlp(rawtext)
		d = []
		skills = []
		if len(doc.ents)==0:
			return render_template("index.html", rawtext=rawtext, message="Pas de compétences révélées dans ce texte !")
		else:
			for ent in doc.ents:
				if nlp(ent.text)[0].pos_ == "VERB":
					entmod = " ".join([str(nlp(ent.text)[0].lemma_)]+[str(token) for token in nlp(ent.text)[1:]])
					d.append((ent.label_, entmod.capitalize()))
					skills.append(entmod)
				else:
					d.append((ent.label_, ent.text.capitalize()))
					skills.append(ent.text)
				resume = ", ".join([x for x in skills]).lower()
				skills_entities = get_skills_entities(pd.DataFrame(d, columns=('named entity', 'output'))) 
			
			job_skills = load_json("data/job_skills.json")
			resumeMatchPercentageJobs = {}
			for j in job_skills:
				job_description =  ", ".join([x for x in j["skills"]]).lower()
				text = [resume, job_description]
				cv = CountVectorizer()
				count_matrix = cv.fit_transform(text)
				matchPercentage = (cosine_similarity(count_matrix)[0][1]*100).round(2)
				resumeMatchPercentageJobs.update({j["job_title"]: matchPercentage})
			
			return render_template("index.html", rawtext=rawtext, skills_entities=skills_entities, resumeMatchPercentageJobs=resumeMatchPercentageJobs)

def get_skills_entities(df):
	skills_entity = {}
	for entit in pd.unique(df["named entity"]):
		skills_entity[entit] = df[df['named entity'] == entit]['output'].tolist()
	return skills_entity

def load_json(jsonfile):
	with open(jsonfile, encoding='utf-8') as f:
		data_json = json.load(f)
	return data_json
	
if __name__ == '__main__':
	app.run(debug=True)