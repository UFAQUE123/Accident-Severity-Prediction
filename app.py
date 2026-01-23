# import all the app dependencies
import numpy as np
import streamlit as st
import joblib

# set the page configuration (MUST BE FIRST)
st.set_page_config(
    page_title="Accident Severity Prediction App",
    page_icon="🛣️",
    layout="wide"
)

# load the encoder and model
models = {
    "Random Forest Classifier": joblib.load("trained_models/rta_model.joblib"),
    "Tuned RF": joblib.load("trained_models/rta_tuned_rf.joblib"),
    "Tuned KNN": joblib.load("trained_models/rta_tuned_knn.joblib"),
    "Tuned DT": joblib.load("trained_models/rta_tuned_dt.joblib")
}
         
encoder = joblib.load("trained_models/ordinal_encoder.joblib")

#creating option list for dropdown menu
day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
collision_type = ['Vehicle with vehicle collision','Collision with roadside objects',
                           'Collision with pedestrians','Rollover','Collision with animals',
                           'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',
                           'Other','With Train']

sex = ['Male','Female','Unknown']

education_level = ['Junior high school','Elementary school','High school',
                           'Unknown','Above high school','Writing & reading','Illiterate']

service_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']

accident_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

# features list
features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day',
            'Type_of_collision','Age_band_of_driver','Sex_of_driver','Educational_level',
            'Service_year_of_vehicle','Day_of_week','Area_accident_occured']

# title
st.markdown(
    "<h1 style='text-align: center;'>🛣️ Accident Severity Prediction App</h1>",
    unsafe_allow_html=True
)

st.divider()
model_choice = st.selectbox("Select Model", ["Random Forest Classifier", "Tuned RF",
                                             "Tuned KNN", "Tuned DT"])
model = models[model_choice]

def main():
    with st.form("road_traffic_severity_form"):
        st.subheader("Please enter the following inputs:")

        No_vehicles = st.slider("Number of vehicles involved:", 1, 7, value=1)
        No_casualties = st.slider("Number of casualties:", 1, 8, value=1)
        Hour = st.slider("Hour of the day:", 0, 23, value=5)

        collision = st.selectbox("Type of collision:", collision_type)
        Age_band = st.selectbox("Driver age group:", age)
        Sex = st.selectbox("Sex of the driver:", sex)
        Education = st.selectbox("Education of driver:", education_level)
        service_vehicle = st.selectbox("Service year of vehicle:", service_year)
        Day_week = st.selectbox("Day of the week:", day)
        Accident_area = st.selectbox("Area of accident:", accident_area)

        submit = st.form_submit_button("Predict")

    if submit:
        input_array = np.array(
            [collision, Age_band, Sex, Education, service_vehicle, Day_week, Accident_area],
            ndmin=2
        )

        encoded_arr = encoder.transform(input_array).ravel().tolist()
        numeric_arr = [No_vehicles, No_casualties, Hour]

        final_input = np.array(numeric_arr + encoded_arr).reshape(1, -1)

        prediction = model.predict(final_input)
        probabilities = model.predict_proba(final_input)

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error(f"🚨 Severity: Fatal Injury (Confidence: {probabilities[0][0]*100:.2f}%)")
        elif prediction[0] == 2:
            st.success(f"✅ Severity: Slight Injury (Confidence: {probabilities[0][2]*100:.2f}%)")
        else:
            st.warning(f"⚠️ Severity: Serious Injury (Confidence: {probabilities[0][1]*100:.2f}%)")

# run app
if __name__ == "__main__":
    main()









