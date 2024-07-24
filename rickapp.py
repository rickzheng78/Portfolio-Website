#Base Packages
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

#Exploratory Data Analysis Packages
import numpy as np
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
import matplotlib.pyplot as plt
import seaborn as sns

# Training and Modeling
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Model Implemetation Package
import joblib 

#Website Configuration
weblogo = Image.open("images/websitelogo.png")
st.set_page_config(page_title='Rick Zheng\'s Portfolio' ,layout="wide",page_icon=weblogo)

#Overall Page
st.title("Rick Zheng")

menu = ['Home Page', 'Text to Sentiment Classifier', 'Calories Burned Predictor']
menu_nav = st.sidebar.selectbox('Menu', menu)

linkedin = {
    'script': """<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="rickkzheng" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://www.linkedin.com/in/rickkzheng?trk=profile-badge"></a></div>
    """
}

with st.sidebar:
        components.html(linkedin['script'], height = 310)

#Home Page
if menu_nav == 'Home Page':

    profpic = Image.open("images/profilepic.jpg")
    st.image(profpic, caption = '--- Taken @ Lizards Mouth Rock ---  Santa Barbara', width = 225)

    st.subheader('About Me')
    st.write("My professional interests lie at the intersection of real-world business problem-solving and data visualization. "
     "I have previous professional and academic experience with statistical analysis, database management, and data-driven decision making. "
     "My current goal is to gain the opportunity to apply my data-driven skills for better change in the workforce.")

    interests, education = st.columns(2)

    with interests:
        st.write("")
        st.subheader("Interests")
        st.write(
            """
            -   Data Mining and Statistical Analysis
            -   Data Visualization and Presentation
            -   Business Intelligence
            """
        )

        st.subheader("Favorite Activities")
        st.write(
            """
            -   Cooking
            -   Travel
            -   Gym
            """
        )

    with education: 
        st.write("")
        st.subheader("Education")
        st.write('University of Michigan - Ann Arbor   \n'
        'Applied Data Science M.S ~ 2024')
        st.write("")
        st.write('University of California - Santa Barbara   \n'
        'Statistics & Data Science B.S ~ 2022')

if menu_nav == 'Text to Sentiment Classifier':

    st.title("Text to Sentiment Classifier")

    sentiment_list = ["Predict Sentiment", 'Exploratory Data Analysis']
    menu_sentiment = st.radio("App Menu", sentiment_list)

    best_model = joblib.load(open("models/Emotion_Classifier_Model.pkl", "rb"))

    def predict_emotions(text):
        results = best_model.predict([text])
        return results

    def get_prediction_proba(text):
        results = best_model.predict_proba([text])
        return results

    emotions_emoji_dict = {"Negative":"😠", "Positive":"🤗"}

    if menu_sentiment == "Exploratory Data Analysis":

        st.write("""Dataset Link: 
        https://www.kaggle.com/datasets/parulpandey/emotion-dataset
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='text-align: center;'> Positive Sentiment Wordcloud </h4>", unsafe_allow_html=True)
            st.image('images/pos.png')

            st.markdown("<h4 style='text-align: left;'> Confusion Matrix </h4>", unsafe_allow_html=True)
            st.write("""Utilizing Accuracy, Precision, and Recall Metrics to compare 
            machine learning models, the Logistic Regression model proved to be overall the best model. 
            The plot below shows its accuracy breakdown.""")
            st.image('images/conf.png')

        with col2:
            st.markdown("<h4 style='text-align: center;'> Negative Sentiment Wordcloud </h4>", unsafe_allow_html=True)
            st.image('images/neg.png')

    if menu_sentiment == "Predict Sentiment":

        with st.form(key = 'Text Input'):
            raw_text = st.text_area("Insert Text Here")
            submit_text = st.form_submit_button(label = 'Begin Analysis')
            
        if submit_text:
            col1, col2 = st.columns(2)

            #Applying Model 
            prediction = predict_emotions(raw_text)

            prob = get_prediction_proba(raw_text)
            format_prob = round(np.max(prob), 2) * 100

            with col1:

                st.success('Prediction')

                emoji_icon = emotions_emoji_dict[prediction[0]]
                st.subheader("{} --- {}".format(prediction[0], emoji_icon))

                st.write("Confidence: {}%".format(format_prob))

            with col2:
                st.success("Sentiment Probabilities")
                prob_df = pd.DataFrame(prob, columns = best_model.classes_).T
                prob_df_cleaned = prob_df.reset_index()
                prob_df_cleaned.columns = ['Sentiment', 'Probability']
                    

                fig = alt.Chart(prob_df_cleaned).mark_bar().encode(
                    x = alt.X('Sentiment',  axis=alt.Axis(labelAngle=0)), 
                    y = 'Probability',
                    color = 'Sentiment'
                )

                st.altair_chart(fig, use_container_width = True)


if menu_nav == 'Calories Burned Predictor':

    st.header('Calories Burned Predictor')

    calories_list = ["Predict Calories", 'Exploratory Data Analysis']
    menu_calories = st.radio("App Menu", calories_list)

    # Preprocessing Data
    calories = pd.read_csv("data/calories.csv")
    exercise = pd.read_csv("data/exercise.csv")

    gym = pd.merge(calories, exercise)

    gym['Weight'] = gym['Weight'] * 2.205
    gym['Height'] = gym['Height'] / 2.54
    gym['Body_Temp'] = gym['Body_Temp'] * (9/5) + 32

    gym['Gender'] = gym['Gender'].str.replace("male", "Male")
    gym['Gender'] = gym['Gender'].str.replace("feMale", "Female")

    gym['Age'] = gym['Age'].astype(float)
    bins = [20, 30, 45, 60, 80]
    bin_labels = ['Early Adulthood', 'Middle Adulthood', 'Late Adulthood', 'Senior']
    gym['Age'] = pd.cut(gym['Age'], bins, labels = bin_labels, right = False)

    char_to_replace = {'Early Adulthood': 'Adulthood', 'Middle Adulthood': 'Adulthood', 
                'Late Adulthood': 'Late Adulthood', 'Senior': 'Late Adulthood'}

    for key, value in char_to_replace.items():
        gym['Age'] = gym['Age'].replace(key, value)

    if menu_calories == 'Predict Calories': 

        sex_list = ['Male', 'Female']
        age_list = ['Adulthood', 'Late Adulthood']

        sex = st.selectbox(label = 'Biological Sex', options = sex_list)
        age = st.selectbox(label = 'Age Group', options = age_list)

        height= st.slider('Height', 50, 85)
        weight = st.slider('Weight', 80, 290)
        duration = st.slider("Work Out Duration", 1, 30)
        heart_rate = st.slider("Heart Rate", 70, 128)
        body_temp = st.slider(label  = "Body Temp", min_value = 98.0, max_value = 106.0, step = 0.5)

        # Feature Engineering

        from sklearn.preprocessing import LabelEncoder

        gender_encoder = LabelEncoder()
        gender_encoder.fit(gym['Gender'])
        gender_values = gender_encoder.transform(gym['Gender'])

        age_encoder = LabelEncoder()
        age_encoder.fit(gym['Age'])
        age_values = age_encoder.transform(gym['Age'])

        gender_df = pd.DataFrame(gender_values, columns = ['Biological Sex'])
        age_df = pd.DataFrame(age_values, columns = ['Age Group'])

        combined = pd.concat([gym, gender_df, age_df], axis = 1)

        df = combined.drop(columns = ['Gender', 'Age'])

        X = df.drop(columns = ['Calories', 'User_ID'])
        Y = df['Calories']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)

        # Implementing Best Performance Model
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(random_state = 123)
        rf.fit(X_train,y_train)  

        def predict_cals(ht, wt, duration, heart_rate, body_temp, sex, age): 

            feature = np.zeros(len(X.columns))

            feature[0] = ht
            feature[1] = wt
            feature[2] = duration
            feature[3] = heart_rate
            feature[4] = body_temp
            if sex == 'Female':
                feature[5] = 0
            if sex == 'Male':
                feature[5] = 1
            if age == 'Adulthood':
                feature[6] = 0
            if age == 'Late Adulthood':
                feature[6] = 1

            return rf.predict([feature])[0]

        if st.button('Predict'):
            st.subheader(predict_cals(height, weight, duration, heart_rate, body_temp, sex, age))
            st.markdown("<h5 style='text-align: left;'> Calories </h5>", unsafe_allow_html=True)

    if menu_calories == 'Exploratory Data Analysis':

        st.write("""
            Dataset Link:
            https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos
            """)

        st.write("""
            Variable Clarifications and Unit of Measurement: 
            -   Adulthood Age Range: (20 - 45)
            -   Late Adulthood Age Range: (46 - 80)
            -   Duration (min)
            -   Height (in)
            -   Weight (lbs)
            -   Body Temp (Fahrenheit)
            -   Heart Rate (bpm)
        """)

        st.markdown("<h4 style='text-align: left;'> Weight v.s Height </h4>", unsafe_allow_html=True)
        st.image('images/ht_wt.png')
        st.markdown('''
            The scatterplot describes a strong positive linear relationship between Weight and Height. Additionally, the color of the points reveals that 
            the distribution of people that identify as Male generally are taller and weigh more than those thatidentify as Female. 
        ''')

        st.markdown("<h4 style='text-align: left;'> Calories Burned Density Distribution </h4>", unsafe_allow_html=True)
        st.image('images/calories_age.png')
        st.markdown('''
            There are overlapping distributions, each representing a different age group. Overall, relative shapes are seen to be similar, with many of the 
            people in this dataset burning close to 30 calories for their workout. 
        ''')

        st.markdown("<h4 style='text-align: left;'> Correlation Matrix </h4>", unsafe_allow_html=True)
        st.image('images/correlation.png')
        st.markdown('''
            Able to see several variables that are significant to the "Calories" response variable (Duration, Heart Rate, Body Temperature). Furthermore, it is 
            good to see the feature variables are hardly related to each other, so there is no sign of multicollinearity. 
        ''')

        compare_models = pd.read_csv("images/compare_models.csv")

        st.markdown("<h4 style='text-align: left;'> Model Testing </h4>", unsafe_allow_html=True)
        st.write(compare_models)
        st.markdown('''
            From the dataframe above, it can be see that the Random Forest model performed the best, having lower error values and a higher R squared values 
            than the other models tested. Therefore, I chose to use Random Forest to fit to my training set to help create the interactive model. 
        ''')




















