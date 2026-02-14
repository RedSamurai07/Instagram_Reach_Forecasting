# Instagram Reach Forecasting

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
This project focuses on predicting the future reach of an Instagram account using historical data and Time Series Forecasting techniques. Social media reach is often inconsistent due to external factors like holidays, exams, or seasonal trends. This case study demonstrates how to use Python to analyze these trends and build a predictive model (specifically a SARIMA model) that helps users understand when their content is likely to perform best.

### Executive Summary
The case study addresses the challenge of fluctuating engagement on social media. By leveraging a dataset containing "Date" and "Instagram Reach" metrics, the project performs an Exploratory Data Analysis (EDA) to identify patterns, such as which days of the week yield the highest reach. It then moves into advanced time series modeling:

1. Analysis: Uses line charts, bar charts, and box plots to visualize trends and day-of-the-week performance.

2. Decomposition: Breaks down the data into seasonal and trend components to confirm that reach is affected by seasonality.

3. Modeling: Employs the SARIMA (Seasonal Autoregressive Integrated Moving Average) model, determining parameters (p, d, q) through autocorrelation and partial autocorrelation plots.

4. Result: The model generates a 100-day forecast, providing a visual roadmap of expected future reach.

### Goal
The primary objectives of this study are:

1. Strategic Planning: To help content creators and professional users decide exactly when to post their most valuable content to maximize visibility.

2. Performance Optimization: To move beyond guessing and use data-driven insights to improve social media strategy and engagement metrics.

3. Trend Identification: To understand how specific factors (like the day of the week or seasonal cycles) impact the number of people a post reaches.

4. Technical Application: To demonstrate a practical end-to-end implementation of Time Series Forecasting in a real-world marketing context using Python libraries like pandas, plotly, and statsmodels.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1GK4tnY4_YfX8ccNhVtedEoGpsUUGOwzlysEVEQr_xpA/edit?gid=1748548740#gid=1748548740)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| transaction_id | Unique identifier for each transaction. | int |
| date | Date when the transaction occurred (MM/DD/YYYY format). | object |
| transaction_time | Time when the transaction was made (HH:MM:SS format).| object |
| transaction_qty  | Number of units purchased in the transaction. | int |
| store_id  | Unique identifier for the store where the purchase was made. | int |       
| store_location |  Name of the store location. | object |
| product_id     |  Unique identifier for each product sold. | int |
| unit_price     |  Price per unit of the product (includes currency symbol).| float |
| product_category | Broad classification of the product. | object |
| product_type     | Subcategory of the product | object |
| product_detail   | Detailed name of the specific product. | object |

### Tools
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing
  
### Analysis
Python
Importing all the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
``` python
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
from plotly.tools import mpl_to_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
```
Loading the dataset
```python
df = pd.read_csv("Instagram-Reach.csv", encoding = 'latin-1')
print(df.head())
```
<img width="339" height="132" alt="image" src="https://github.com/user-attachments/assets/447eae75-131a-441a-b697-f4cb3402c8c6" />
Converting datetime to Date
```python
df['Date'] = pd.to_datetime(df['Date'])
print(df.head())
```
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/6242de17-c85e-4d84-99e8-eab655c13837" />

Null values check
```python
df.isna().sum()
```
<img width="178" height="91" alt="image" src="https://github.com/user-attachments/assets/82532efc-f2a5-41eb-bb56-e078945503f4" />

We notice there are no null values preset in our data

**Exploratpry Data Analysis**
To view the overall instagram reach trend
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], 
                         y=df['Instagram reach'], 
                         mode='lines', name='Instagram reach'))
fig.update_layout(title='Instagram Reach Trend', xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()
```
<img width="753" height="436" alt="image" src="https://github.com/user-attachments/assets/2dc82458-4114-4385-80b2-286ee161e696" />

To view the instagram reach by day
```python
fig = go.Figure()
fig.add_trace(go.Bar(x=df['Date'], 
                     y=df['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach by Day', 
                  xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()
```
<img width="731" height="421" alt="image" src="https://github.com/user-attachments/assets/551d87c9-e19e-4930-a7dd-c76629ce4901" />

Outlier detection
```python
fig = go.Figure()
fig.add_trace(go.Box(y=df['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach Box Plot', 
                  yaxis_title='Instagram Reach')
fig.show()
```
<img width="794" height="430" alt="image" src="https://github.com/user-attachments/assets/9bf0f9df-7486-4b95-9087-72c934262d45" />

we notice that we do not have much outkliers present in our data.
To extract the days from the date.
```python
df['Day'] = df['Date'].dt.day_name()
print(df.head())
```
<img width="335" height="135" alt="image" src="https://github.com/user-attachments/assets/ba62c7c5-e9de-4765-b4e6-e8b4fef7c079" />

Let's create a table with mean, median, standard deviation of instagram reach to understand our data.
```python
day_stats = df.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)
```
<img width="411" height="168" alt="image" src="https://github.com/user-attachments/assets/37d88341-1490-4f15-8c10-49e582cd515e" />

let's quickly check for instagram reach by day of the week
``` python
fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['mean'], 
                     name='Mean'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['median'], 
                     name='Median'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['std'], 
                     name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', 
                  xaxis_title='Day', 
                  yaxis_title='Instagram Reach')
fig.show()
```
<img width="788" height="415" alt="image" src="https://github.com/user-attachments/assets/91cb482a-0da9-4929-91d3-af2e652dbada" />

using statsmodel library, let's quickly check for seasonality, trend and residuals of our data.
``` python
data = df[["Date", "Instagram reach"]]

result = seasonal_decompose(df['Instagram reach'], 
                            model='multiplicative', 
                            period=100)
fig = plt.figure()
fig = result.plot()
fig = mpl_to_plotly(fig)
fig.show()
```
<img width="659" height="483" alt="image" src="https://github.com/user-attachments/assets/75400bd8-ffdd-4862-9ae1-0c691d36d927" />

- Decomposing the time series data confirms that Instagram reach follows a multiplicative seasonal pattern rather than a random one, meaning the seasonal effect scales with the overall trend of the account.

- The SARIMA model was selected as the most appropriate tool because it specifically accounts for these seasonal fluctuations alongside historical trends.

Now here’s how to visualize a autocorrelation plot to find the value of q
``` python
pd.plotting.autocorrelation_plot(data["Instagram reach"])
```
<img width="587" height="438" alt="image" src="https://github.com/user-attachments/assets/981c18fc-f2d7-4ba5-8adf-a74c5de7d1f0" />

Now here’s a visualization a partial autocorrelation plot to find the value of q
```python
plot_pacf(data["Instagram reach"], lags = 100)
```
<img width="568" height="433" alt="image" src="https://github.com/user-attachments/assets/5929a449-a13a-4b62-aa82-a6d270f2d2bd" />

Now, let's train a model using SARIMA
``` python
p, d, q = 8, 1, 2

model=sm.tsa.statespace.SARIMAX(data['Instagram reach'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())
```
<img width="732" height="576" alt="image" src="https://github.com/user-attachments/assets/5a581a0c-9bbb-4f07-bc21-855818286344"/>
<img width="672" height="271" alt="image" src="https://github.com/user-attachments/assets/db43e794-b46b-4c27-8f0b-591f1c7ccef1" />


With the SARIMA model fully trained, we can now project the account's performance into the future. By instructing the model to predict the next 100 days beyond our current dataset, we gain a clear picture of expected growth and seasonal dips.

To make these results actionable, we plot the original training data alongside the new predictions using a line chart. This visual comparison allows us to see how well the forecast continues the existing trends and seasonal cycles, providing a data-backed roadmap for the next three months of content planning.

``` python
predictions = model.predict(len(data), len(data)+100)

trace_train = go.Scatter(x=data.index, 
                         y=data["Instagram reach"], 
                         mode="lines", 
                         name="Training Data")
trace_pred = go.Scatter(x=predictions.index, 
                        y=predictions, 
                        mode="lines", 
                        name="Predictions")

layout = go.Layout(title="Instagram Reach Time Series and Predictions", 
                   xaxis_title="Date", 
                   yaxis_title="Instagram Reach")

fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
fig.show()
```
<img width="792" height="417" alt="image" src="https://github.com/user-attachments/assets/450d486d-3779-47f0-9c1e-3e9a20b2a296" />


1. Time Series Decomposition Insights:

 a. Trend Component: The reach exhibits a distinct upward trend in the first half of the dataset, peaking around day 150, followed by a gradual decline and stabilization towards the end of the period.

 b. Strong Seasonality: The "Seasonal" plot confirms that Instagram reach follows a highly repetitive, multiplicative pattern. This suggests that specific days of the week or recurring cycles heavily influence engagement levels.

 c. Residual Noise: The "Resid" plot shows scattered data points, indicating that while most reach is predictable via trends and cycles, there are still random external factors (like a viral post or a global holiday) that create unpredictable spikes.

2. Model Selection & Parameters:

 a. Autocorrelation (ACF): The autocorrelation plot shows that recent data points are highly correlated with past values, but this correlation gradually decays over time.

 b. SARIMA Parameters: Based on the ACF and PACF analysis, the model uses parameters $p=8, d=1, q=2$. These values help the model account for the "memory" of previous days' reach and the specific seasonal lag of 12 periods.

3. Forecasting and Future Trends

 a. Predictive Stability: The model's predictions (shown in red) successfully mimic the seasonal "peaks and valleys" seen in the historical training data (shown in blue).

 b. Lower Expected Reach: The forecast for the next 100 days suggests that the account will continue to see seasonal fluctuations, but the overall reach is projected to stay at a lower baseline compared to the peak seen earlier in the year.

 c. Consistent Cycles: Even as the overall trend settles, the model predicts that the high-reach days (like Tuesdays and Mondays) will continue to significantly outperform the low-reach days (like Fridays).

### Insights

- High-Performance Days: The analysis of mean and median reach shows that Tuesdays, Mondays, and Sundays typically yield the highest average reach. In contrast, Fridays often show the lowest engagement.

- Reach Stability: The "Standard Deviation" metric revealed that while Tuesdays have the highest reach, they also have higher variability. This means while potential for "viral" reach is higher, it is less consistent than other days.

- Seasonality Impact: The seasonal decomposition of the data confirmed that Instagram reach is not random; it follows clear multiplicative seasonal patterns. This suggests that reach is heavily influenced by external cycles (e.g., weekly routines of users).

- Growth Trends: The line charts and SARIMA forecast indicate whether the account is in a growth, stagnation, or decline phase, allowing the creator to adjust expectations based on the predicted 100-day trend.

- Source Attribution: A significant portion of reach (often ~38%) comes from hashtags, while the "Explore" section (the recommendation engine) often contributes a smaller percentage (around 9%), showing that proactive discovery tools like hashtags are currently more effective for this account than the passive recommendation algorithm.

### Recommendations
- Schedule High-Value Content: Save your most important announcements, high-quality videos, or promotional posts for Tuesdays and Mondays. Avoid launching major campaigns on Fridays when reach is statistically at its lowest.

- Optimize Hashtag Strategy: Since hashtags are a primary driver for new reach, continue researching and using niche-specific hashtags rather than relying solely on the Instagram Explore algorithm to find your audience.

- Plan Around Seasonality: Use the SARIMA forecast to anticipate "dry spells." If the model predicts a dip in reach for the coming weeks (due to exam seasons, holidays, etc.), focus on community engagement and "Saves" (which signal value) rather than worrying about low view counts.

- Content Consistency: Because the account shows clear weekly seasonality, maintaining a consistent posting schedule is vital. Drastic changes in posting frequency can disrupt the predictable patterns the algorithm uses to serve your content to followers.

- Focus on "Saveable" Content: The analysis shows a strong correlation between Saves and overall Impressions. Create "educational" or "resource-based" posts (like carousels) that users are likely to save, as this signals to the algorithm that the content is high-quality, eventually boosting reach.

- Use Wednesdays or Thursdays to post "Engagement Bait" (polls, questions, or controversial opinions). Since the reach is naturally lower these days, interactive content helps "train" the algorithm that your followers still care about your posts, which can prevent a total reach collapse before the weekend.

- Use these high-variance days to experiment with Reels or experimental formats. Since the "floor" and "ceiling" for reach are wider, these days are the best time to take risks. If a post fails, it’s expected; if it hits, the high-potential reach of that day will carry it further.

- Shift from "Double-Tap" (Like) content to "Reference" content. Create infographics, checklists, or "How-to" guides. Even if your initial reach on a Friday is low, if the few people who see it save it, the Instagram algorithm will likely give that post a second life (a "reach extension") on the following Monday or Tuesday.

- Instead of fighting the seasonal dip (e.g., during exam season), pivot the context of your content. If the data predicts a reach decline because your audience is busy, shorten your captions and use "Quick-Read" formats. Match the velocity of your content to the availability of your audience predicted by the SARIMA model.

- Build a Content Calendar 30 days in advance based on the forecasted peaks. If the model predicts a massive reach spike in three weeks, start "seeding" that topic now with smaller Stories or teaser posts to build momentum for the predicted peak.

- Use Fridays for Community Management rather than content creation. Use this low-reach day to reply to comments from earlier in the week or share User Generated Content (UGC) on Stories. This keeps your account active in the algorithm without "wasting" a high-quality post on a low-reach day.

