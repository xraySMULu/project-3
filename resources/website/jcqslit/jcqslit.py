import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import configparser as config
import altair as alt


st.set_page_config(
    page_title="California Living",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar Navigation
with st.sidebar:
   st.sidebar.image("/Users/jacintocepedaquroz/jcqslit/calist.png", width=160)
   st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to Model", ["Home", "Home Pricing Based On ROI", "Interest Predictions","Home Feature Analysis", "California Investor Recommendation"])
        # Home Page
if page == "Home":
       st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="margin: 0;">Golden State Housing 🌉</h1>
        <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F544e2af6-1620-4f6e-a6a7-c7604d5aa54a_800x450.gif" width="400">
    </div>
    """,
    unsafe_allow_html=True
)
       st.markdown(
    r""" 
    
    ## Project Overview

    The primary goal of our project, Golden State Housing Insights, is to predict housing prices in the state of California. 
     Our team aims to achieve this by leveraging machine learning models to analyze various factors, including investor return on investment (ROI), affordability, distance between homes and cities, and specific home features.
     By integrating these elements, we strive to provide accurate and actionable insights into the California housing market, aiding investors, homebuyers, and policymakers in making informed decisions.
   ☀️ 
    ## Business Scenario 
    
      We approached our business problem as independent real estate consultants. Our client, who recently accepted a job offer in California, seeks to purchase a home in the area. They have tasked us with identifying the top 5 metro areas with the highest return on investment (ROI) based on a home feature analysis. Additionally, we will predict near-term interest rates using Time Series modeling and linear regression to support their decision-making.
    
      ## Model Training & Testing

      #### - **Housing Price Prediction using Zillow Data**

      Our team employed a comprehensive data pre-processing approach to ensure the accuracy and reliability of our housing price predictions. We utilized powerful libraries such as NumPy and Pandas for efficient data manipulation and analysis. Matplotlib was used for visualizing data trends and patterns. We also applied data melting techniques to reshape our datasets, making them more suitable for analysis. Additionally, we incorporated time series analysis to account for temporal trends and seasonality in housing prices. This robust pre-processing framework enabled us to prepare our data effectively for machine learning modeling. Our team conducted an extensive exploratory data analysis (EDA) to uncover underlying patterns and relationships within the housing data. This initial step allowed us to gain valuable insights and informed our subsequent modeling approach.     
      
      #### - **Interest Rate Prediction**

      To predict future interest rates up to February 2025, the code utilizes historical data from a CSV file ('fed-rates.csv'). The data spans from January 2017 to December 2024, and is cleaned to calculate average monthly rates.

      
      #### - **Housing Feature**

      To prepare the dataset for analysis, we began by removing outliers by filtering extreme values in the features AveRooms, AveBedrms, Population, and AveOccup. We then used a heatmap to identify the most impactful variables for predicting house prices. The data was split into training and testing sets, with 80% allocated for training and 20% for testing using the train_test_split() function. Finally, we standardized the values using StandardScaler() to ensure consistent scaling across features.
      To predict housing market trends and analyze influential home features, we utilized the Ames dataset, assuming Californians have similar preferences. We began by preprocessing the dataset, removing features with sparse data and encoding non-numerical features. A correlation analysis was then conducted to identify the 12-15 most influential home features based on their correlation with home prices.
      
      """
)

#Page on ROI 
elif page== "Home Pricing Based On ROI":
         st.title("California Home Pricing Based On ROI 💰")
         st.write("Results of the predictive model created by Chris Gilbert")
         #Show CG Demo
         st.subheader("Arima Model Demo")
         st.video("/Users/jacintocepedaquroz/jcqslit/cgdemo.mp4")
         #Load CSV file
         file_path = "/Users/jacintocepedaquroz/jcqslit/caliroi.csv"
         roi_df = pd.read_csv(file_path)
         st.subheader("2025 ARIMA Prediction Results Table")
         st.dataframe(roi_df, use_container_width=True)
         roi_csv = roi_df.to_csv(index=False).encode("utf-8")
         st.download_button("Download CSV", roi_csv, "roi_predictions.csv", "text/csv", key="download-csv")
         #Visuals for Top 5 Metro Areas
         st.subheader("The Top Five Metro Areas By ARIMA")
         st.image("/Users/jacintocepedaquroz/Desktop/CG.Visuals/Merced.png")
         st.image("/Users/jacintocepedaquroz/Desktop/CG.Visuals/Modesto.png")
         st.image("/Users/jacintocepedaquroz/Desktop/CG.Visuals/Sacramento.png")
         st.image("/Users/jacintocepedaquroz/Desktop/CG.Visuals/Bakersfield.png")
         st.image("/Users/jacintocepedaquroz/Desktop/CG.Visuals/Oxnard.png")

         ## Housing Price Prediction using Zillow Data Analysis
         st.subheader("Housing Price Prediction using Zillow Data Analysis" )
         st.markdown(r"""To identify the five most optimal metro areas for investment in California, we automated the process of running time series models (ARIMA and SARIMA) for each of the 34 metro areas. This automation was necessary due to the impracticality of manually analyzing each area. We evaluated the models' accuracy by comparing their predictions for December 2024 with actual observed values. The ARIMA model's predictions were **9%** off observed values with **91%** accuracy, while the SARIMA model's predictions were **14%** off observed values with **86%** accuracy.
                           Despite SARIMA showing better ROI for the top 5 metro areas, ARIMA's lower error rate suggests it could be more accurate with additional data. The ARIMA model predicted a return on investment (ROI) percentage range of **3%** to **6%**, whereas the SARIMA model predicted an ROI a of **6%** . For model optimization, both ARIMA and SARIMA had high RMSE values, **2432.90** and **3547.65** respectively. A model with a high RMSE has more error and less precise predictions, as this value should ideally be below **100**. """)
         
#Page of interest Rate Predictions
elif page== "Interest Predictions":
         st.title("Interest Predictions 📈")
         st.write("Interest rate predictive model created by Dexter Johnson")
  
         #Loads Demo
         st.write("### Interest Rate Predicition Model Demo")
         st.video("dexdemo.mp4")
         #Loads Images 
         st.subheader("Actual vs. Predicted Interested Rates(KNN)")
         st.image("/Users/jacintocepedaquroz/jcqslit/dex_lr2.png")
         st.markdown(r""" 
- RMSE (Root Mean Squared Error): This measures the average difference between the predicted and actual interest rates. A lower RMSE value indicates a better model fit. I obtained an RMSE of **0.106**.

- MAE (Mean Absolute Error): Similar to RMSE, but less sensitive to large errors, MAE provides another perspective on prediction accuracy. My MAE was **0.06**.

- R-squared (R²): This metric represents how well the model explains the variation in interest rates. It ranges from 0 to 1, with higher values suggesting a better fit. I achieved an R² score of **0.99**.
""")
         st.subheader("Interest Rate Predictions Until Feb 2025")
         st.image("/Users/jacintocepedaquroz/jcqslit/dex_lr1.png")
         #Load CSV File 
         file_path_2 = "/Users/jacintocepedaquroz/jcqslit/future_predictions.csv"
         fp_df = pd.read_csv(file_path_2)
         st.write("### Interest Rate Predictions 2025")
         st.dataframe(fp_df)
         fp_csv = fp_df.to_csv(index=False).encode("utf=8")
         st.download_button("Download CSV", fp_csv, "future_predictions.csv", "text/csv", key= "download-csv")
         st.subheader("Interest Prediction Analysis")
         st.markdown(r""" The models aim to predict future interest rates based on past trends. Assuming the historical patterns continue, the models could potentially forecast whether interest rates are likely to increase, decrease, or remain stable in the near future (e.g., the next few months). From the code's plot titled "Interest Rate Prediction until Feb 2025," we see an upward trend in interest rates over the past years leading to now. In this scenario, we would generally anticipate a slowdown in the housing market with potentially decreased sales and slower home price appreciation. This model predicts an increase in the interest rates into Feb 2025. This is primarily because interest rates directly affect mortgage rates. Higher mortgage rates make homes less affordable, leading to decreased demand and potentially lower home prices. Lower mortgage rates make homes more affordable, potentially increasing demand and driving up home prices.             
         """ )

#Page of Predicting California House Pricing Using Features
elif page == "Home Feature Analysis": 
         st.title("Predicting California House Pricing using Features 🏡")
         st.write("Predictive housing model created by Roderick Burroughs and Will Atwater")
         st.subheader("Predicting Home Prices Using Home Features Demo")
         st.video("/Users/jacintocepedaquroz/jcqslit/rb.mp4")
         st.subheader("Feature Correlation Heatmap")
         st.image("/Users/jacintocepedaquroz/jcqslit/rb_heatmap.png")
         st.write(
         "The findings from the analysis indicate that the five most influential features in predicting house prices are Median Income (MedInc), House Age (HouseAge), and Average Number of Rooms (AveRooms), Latitude, and Average Occupancy . Among these, Median Income exhibits the strongest correlation with housing prices.")
         st.subheader("Linear Regression: Actual vs Predicted House Prices")
         st.image("/Users/jacintocepedaquroz/jcqslit/rb_linearreg.png")
         st.write("The model, trained using linear regression, and the aforementioned features achieved  an accuracy score of approximately 63%. The scatter plot comparing actual and predicted prices using the 3 features above, shows a general alignment, with some deviations suggesting  a reasonable, though not perfect, predictive capability. More prominent features will be needed.")
         st.subheader("Random Forest Mode")
         st.image("/Users/jacintocepedaquroz/jcqslit/Random_forest.PNG")
         st.write("We then tuned a Random Forest model to see if we could improve accuracy. It resulted in an accuracy of 70.43%. This suggests that there is some economic correlation with housing prices, though not strong enough for reliable predictions of the market, on it's own.")
         st.subheader("Housing Feature Analysis")
         st.markdown(r""" The analysis revealed that the five most influential factors in predicting house prices are Median Income (MedInc), House Age (HouseAge), and Average Number of Rooms (AveRooms), Latitude, and Average Occupancity(AveOccup). Among these, Median Income exhibits the strongest correlation with housing prices, suggesting that areas with higher median incomes tend to have more expensive homes. The model, trained using linear regression, achieved an R-squared (R²) score of approximately 63%. Even after refinment using hypertuning and comparing the Random Forest Model the maximum accuracy score topped at 70.47%. These findings highlight that there is an impact of economic factors on housing prices though not wholly reliable. Access to more impactful features will be needed. Additonally, exterior quality is a very important feature, followed by Garage size and most of the living space size features. Kitchen quality is right behind exterior quality and the overall size of the house.""")

#Conclusion Page
elif page == "California Investor Recommendation":
         st.title("California Investor Explanation & Recommendations 🧠")
         st.image("/Users/jacintocepedaquroz/jcqslit/gs1.jpg")
         st.markdown(r"""
         
         ### Client Recommendations
            
         Top 5 California Metro Areas with the highest ROI and best potential for a solid investment
         - Merced, CA - ROI @ 1 year - 6%
         - Modesto, CA - ROI @ 1 year - 5%
         - Santa Cruz, CA - ROI @ 1 year - 4%
         - Sacramento, CA - ROI @ 1 year - 4%
         - Bakersfield, CA - ROI @ 1 year - 4%

         
         ### Features To Consider
         - House age
         - Exterior Quality
         - Garage Size
         - Living space size
         - Kitchen quality
         
         ### Model Updates
         The prediction percentages come from models trained on five years of data to predict two years ahead. We plan to use ten years of training data to predict just one year ahead. This suggests that our future predictions will likely be even more accurate than those in this validation test. By leveraging a larger dataset, we aim to enhance the reliability and precision of our investment recommendations.
                     
         """ )
         
         people = [
    {"name": "Dexter Johnson", "image": "/Users/jacintocepedaquroz/jcqslit/decover.png"},
    {"name": "Chris Gilbert", "image": "/Users/jacintocepedaquroz/jcqslit/cgcover.png"},
    {"name": "Joel Freeman", "image": "/Users/jacintocepedaquroz/jcqslit/joelcover.png"},
    {"name": "Jacinto Cepeda Quiroz", "image":"/Users/jacintocepedaquroz/jcqslit/jcqcover.png"},
    {"name": "Sean Burroughs", "image":"/Users/jacintocepedaquroz/jcqslit/SB.png"},
    {"name": "Will Atwater" , "image":"/Users/jacintocepedaquroz/jcqslit/WA.png"}
]
         
         st.subheader("California Real Estate Gurus")
         for person in people:
         
          col1, col2 = st.columns([1, 4])  # Column layout (image on left, name on right)
          with col1:
            st.image(person["image"], width=100)  # Adjust width as needed
          with col2:
            st.subheader(person["name"])
         


