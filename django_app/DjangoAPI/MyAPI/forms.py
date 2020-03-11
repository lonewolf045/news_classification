from django import forms

class NewsForm(forms.Form):
	number=forms.IntegerField()
	headline=forms.CharField(max_length=100)
	name=forms.CharField(max_length=20)
	date=forms.CharField(max_length=10)
	content=forms.CharField(max_length=1000)
