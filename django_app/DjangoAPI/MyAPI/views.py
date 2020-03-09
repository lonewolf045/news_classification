from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import gensim
import json

from . models import news
import pickle
from sklearn.externals import joblib
from nltk import word_tokenize

@api_view(["POST"])

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

