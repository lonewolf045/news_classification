from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.contrib import messages
import gensim
import json

from . models import news
from . forms import NewsForm
import pickle
from sklearn.externals import joblib
from nltk import word_tokenize

#@api_view(["POST"])

def newsTag(request):
	try:
		model=joblib.load('/home/sayantan/rough/ml_project/d2v.pkl')
		logreg=joblib.load('/home/sayantan/rough/ml_project/logreg.pkl')

		mydata=request.data
		l=list(mydata.values())
		a=l[4]
		b=word_tokenize(a.lower())
		fv=model.infer_vector(b)

		p=logreg.predict([fv])

		array=['business','entertainment','politics','sports','tech']

		#print(array[int(p[0])])
		return JsonResponse(array[int(p[0])],safe=False)


	except ValueError as e:
		return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

def FormAPI(request):
	if(request.method=='POST'):
		form=NewsForm(request.POST)
		if(form.is_valid()):
			Number=form.cleaned_data['number']
			Headline=form.cleaned_data['headline']
			Name=form.cleaned_data['name']
			Date=form.cleaned_data['date']
			Content=form.cleaned_data['content']
			print(Number,Headline,Name,Date,Content)
			print(type(Content))

			model=joblib.load('/home/sayantan/rough/ml_project/d2v.pkl')
			logreg=joblib.load('/home/sayantan/rough/ml_project/logreg.pkl')

			b=word_tokenize(Content.lower())
			fv=model.infer_vector(b)

			p=logreg.predict([fv])

			array=['business','entertainment','politics','sports','tech']

			res=array[int(p[0])]

			messages.success(request,'Application Status: {}'.format(res))
			#return JsonResponse(array[int(p[0])],safe=False)

	form=NewsForm()

	return render(request, 'Myform/nform.html', {'form':form})




