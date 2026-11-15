from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import sklearn

st.set_page_config(page_title='California Housing | Price Predictor', page_icon='🏠', layout='wide')
FEATURES = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']

@st.cache_resource
def load_model():
    # Load only the trusted model bundled with this internship project.
    return joblib.load(Path(__file__).with_name('house_price_model.joblib'))

st.title('California Housing | Price Predictor')
st.caption('MAINCRAFTS TECHNOLOGY · AI/ML INTERNSHIP · TASK 1')
st.write('Estimate the median house value of a California census block group using a trained linear regression model.')
a,b,c=st.columns(3)
a.metric('Test MAE', '$53,320'); b.metric('Test RMSE', '$74,558'); c.metric('Test R²', '0.5758')
st.divider()
if sklearn.__version__ != '1.6.1':
    st.error('This saved model requires scikit-learn 1.6.1. Install the pinned requirements before running the app.')
    st.stop()
try:
    model=load_model()
except Exception as exc:
    st.error(f'Model could not be loaded: {exc}')
    st.stop()
left,right=st.columns([3,2],gap='large')
with left:
    st.subheader('Housing characteristics')
    with st.form('housing'):
        a,b=st.columns(2)
        with a:
            income=st.number_input('Median household income ($)',1000,150000,35000,step=500)
            age=st.number_input('Median house age (years)',1,52,25)
            rooms=st.number_input('Average rooms per household',0.1,150.0,5.0,step=0.1)
            bedrooms=st.number_input('Average bedrooms per household',0.1,35.0,1.0,step=0.1)
        with b:
            population=st.number_input('Block-group population',1,35682,1200)
            occupancy=st.number_input('Average people per household',0.1,1250.0,3.0,step=0.1)
            latitude=st.number_input('Latitude',32.54,41.95,34.0,step=0.01,format='%.2f')
            longitude=st.number_input('Longitude',-124.35,-114.31,-118.0,step=0.01,format='%.2f')
        submitted=st.form_submit_button('Estimate house value',type='primary',use_container_width=True)
with right:
    st.subheader('Prediction')
    if submitted:
        if bedrooms > rooms:
            st.error('Average bedrooms cannot exceed average total rooms.')
        else:
            # MedInc is measured in $10,000; the target is measured in $100,000.
            row=pd.DataFrame([[income/10000,age,rooms,bedrooms,population,occupancy,latitude,longitude]],columns=FEATURES)
            value=float(model.predict(row)[0])*100000
            st.metric('Estimated median house value', f'${value:,.0f}')
            if value < 0 or value > 500001:
                st.warning('Prediction is outside the observed target range; interpret with extra caution.')
            st.caption('Historical estimate, not a current market valuation or an individual home appraisal.')
    else:
        st.info('Enter the housing characteristics and select Estimate house value.')
    st.markdown('**About the model**')
    st.write('20,640 observations · 8 predictors · 80/20 split · median imputation · ordinary least squares regression.')
    st.write('The original data is from the 1990 California census. Target values were capped near $500,000. Linear regression can produce unrealistic estimates.')
st.divider()
st.caption('Educational project · N. Srinikethan · Maincrafts AI/ML Task 1')
