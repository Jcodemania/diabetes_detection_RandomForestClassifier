#Description: This program detects if someone has diabetes using machine learning and Python!

#Importing Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


#Creating a title and a sub-title
st.write("""
# Diabetes Detection...
Detect if someone has diabetes using Machine Learning and Pyhton!
""")

#Open and display an image
#image=Image.open()
image=Image.open('dd.jpg')
image_style = {'width': '100%', 'height':'70%', 'border': '1px solid black'}
st.image(image)

#Get the data
df=pd.read_csv('diabetes_csv.csv')
#Set a subheader
st.subheader("Data Information")
#Show the data as a table
st.dataframe(df)
#Show statistics on data
st.write(df.describe())
#Show the data as a chart
chart=st.bar_chart(df)

#Split the data into independent 'x' ad dependent 'y' variables
x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values
#Split the data set into 75% Training and 25% Testing
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25, random_state=0)

#Get the feature input from the user
def get_user_input():
	pregnancies=st.sidebar.slider('pregnancies',0,17,3)
	glucose=st.sidebar.slider('glucose',0,199,117)
	blood_pressure=st.sidebar.slider('blood_pressure',0,122,72)
	skin_thickness=st.sidebar.slider('skin_thickness',0,99,23)
	insulin=st.sidebar.slider('insulin',0.0,846.0,30.5000)
	bmi=st.sidebar.slider('bmi',0.0,67.1000,32.0)
	dpf=st.sidebar.slider('dpf',0.0780,2.4200,0.3725)
	age=st.sidebar.slider('age',21,81,29)

	#Store a dictionary into a variable
	user_data={'pregnancies':pregnancies,
		  'glucose':glucose,
		  'blood_pressure':blood_pressure,
		  'skin_thickness':skin_thickness,
		  'insulin':insulin,
		  'bmi':bmi,
		  'dpf':dpf,
		  'age':age
		}
	#Transform the data into a data frame
	features=pd.DataFrame(user_data,index=[0])
	return features
#Store the user input into a variable
user_input=get_user_input()

#Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

#Store the models predictions in a variable
prediction=RandomForestClassifier.predict(user_input)

#Set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)
