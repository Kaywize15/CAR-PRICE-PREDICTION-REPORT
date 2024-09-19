
# CAR PRICE PREDICTION REPORT

# PROBLEM STATEMENT
Developing a predictive model that accurately estimates the market price of a used or new car based on its attributes, such as manufacturer, make, production year, engine volume, gear box type and other relevant features, to help buyers, sellers, and dealers make informed decisions.
#### Specific Goals:
- Explore Data and Date cleaning 
- Analyse and Data Preprocessing to handle missing values, duplicate values, outliers and feature scaling
- Data Transformation
- Develop a model that can generalize well across different car models, brands and regions.
#### Target Variable:
- Price (Continous)
#### Data Requirement:
- Dataset containing at least 20,000 records of cars with their attributes and prices.
- Data from reputable sources such as car dealership and online marketplace.
#### Evaluation Metrics:
- Mean Squared Error (MSE)
- Coeeficient of Determination (R-Squared)

This problem statement provides a clear direction for developing a car price prediction model that can be useful for various stakeholders in the automobile industry.

# LIBRARY USED 
- Pandas
- Numpy
- Matplotlib
- Datetime
- Seaborn
- Sklearn Models
- Warnings

# STEPS

#### Step 1: Importing Libraries
![Screenshot (12)](https://github.com/user-attachments/assets/5c752f98-17c5-4518-8661-fd03eb33a0bd)

#### Step 2: Importing the Dataset
![Screenshot (13)](https://github.com/user-attachments/assets/d44fafb6-c045-4e70-9ad4-3b6fc4827b3d)

#### Step 3: Dataset Columns
![Screenshot (14)](https://github.com/user-attachments/assets/26446589-4057-4db0-bc4a-596d9c16e3de)

#### Step 4: Dataset Description 
![Screenshot (15)](https://github.com/user-attachments/assets/da1d1456-e297-4a05-9ffe-09858d864ea3)

#### Step 5: Checking for Duplicates and Removing Duplicated values
![Screenshot (16)](https://github.com/user-attachments/assets/9f9ff7bd-1a02-425c-93a1-9b09a698c055)

### ANALYSE AND DATA PREPROCESSING

#### Step 6: Unique Number of items in each Columns
![Screenshot (17)](https://github.com/user-attachments/assets/f32ff897-1258-4362-88af-557616366983)

#### Step 7: Dropping Columns - "ID" and "Doors" columns are not in the right data format 
![Screenshot (18)](https://github.com/user-attachments/assets/eaaa22d0-7662-400f-8ed5-d141092cc9ff)

#### Step 8: Replacing "Levy" Column and converting to an integer type format 
![Screenshot (19)](https://github.com/user-attachments/assets/321fd13e-0d99-4006-b1f9-8b1607c917f3)

#### Step 9: Replacing "Engine Volume" column and converting to a float type format
![Screenshot (20)](https://github.com/user-attachments/assets/f3f86211-57ed-455f-b331-d2918aea8f72)

#### Step 10: Removing km from "Mileage" column and converting to an Integer type format
![Screenshot (21)](https://github.com/user-attachments/assets/064aacab-9c58-421e-ac75-055af8d72d24)

#### Step 11: Dealing with Date time by assigning a variable to the function
![Screenshot (22)](https://github.com/user-attachments/assets/f888f899-9e12-4a72-a4f5-ec629ca3e519)

#### Step 12: Replacing the "Prod. year" column by creating a new column "Age" and the function to get informations for the column, also dropping the Prod. year column
![Screenshot (25)](https://github.com/user-attachments/assets/36c85dc0-5875-4835-a854-b081a9d11bb5)

#### Step 13: Checking the Dataset informations
![Screenshot (26)](https://github.com/user-attachments/assets/07186ca5-bad7-4a26-b96e-fe73b0034688)

#### Step 14: Visualization of the Integer columns of the Dataset using Histogram
![Screenshot (27)](https://github.com/user-attachments/assets/f708f2a9-a63a-40e6-b87c-f5bbbce4626f)
![Screenshot (28)](https://github.com/user-attachments/assets/eee8b176-ec3a-4501-b20d-ac8515177859)

#### Step 15: Getting the Top10 car manufacturers from the "Manufacturer" column
![Screenshot (29)](https://github.com/user-attachments/assets/30612719-19bb-4755-b7c8-28cdfc9dc98d)

#### Step 16: Getting the Top10 Mean Price for the car manufacturers.
![Screenshot (30)](https://github.com/user-attachments/assets/bbcc26d2-3bef-47b6-b837-2761e4f1d087)

#### Step 17: Visualizing the Average mean price of the Top10 Car Manufacturers using Matplotlib library
![Screenshot (31)](https://github.com/user-attachments/assets/e73ca9cd-b090-40a0-948b-28f56f21df6e)

#### Step 18: Vizualization for the Count of each "Category" column using Seaborn.
![Screenshot (32)](https://github.com/user-attachments/assets/d704250d-03f4-4ba0-8ec8-8a17bb5b6d75)

#### Step 19: Assigning a variable for object columns for Visualization
![Screenshot (33)](https://github.com/user-attachments/assets/0f2df25d-c234-4382-875d-b2ad0e9714d3)

#### Step 20: Visualization of the Object columns using Matplotlib.
![Screenshot (35)](https://github.com/user-attachments/assets/ccd5d800-f03b-44e7-9aae-ed36f47ad6fa)
![Screenshot (44)](https://github.com/user-attachments/assets/0ae43eba-46b4-43e0-a2e7-6fb952648b6a)
![Screenshot (36)](https://github.com/user-attachments/assets/9bc04ae3-0eb5-44d7-ab1a-8135151dfe7e)
![Screenshot (37)](https://github.com/user-attachments/assets/f12c0c60-6d5f-419d-8495-74957f1748fc)
![Screenshot (38)](https://github.com/user-attachments/assets/dd304570-6336-4e98-965c-3c8439bbd615)
![Screenshot (39)](https://github.com/user-attachments/assets/e44d1030-23fd-4210-9069-042e3237aaac)
![Screenshot (40)](https://github.com/user-attachments/assets/6c7152da-4780-41e9-ad63-46625e8966a1)
![Screenshot (41)](https://github.com/user-attachments/assets/8b0fe693-ca5e-4dc6-93c3-97855c6c13ac)
![Screenshot (42)](https://github.com/user-attachments/assets/23785032-468f-47b7-a8f6-a0132c516aa9)
![Screenshot (43)](https://github.com/user-attachments/assets/9e70648c-a163-49ca-b1a2-7c64272031af)

#### Step 21: Dataset Correlation excluding the object columns.
![Screenshot (45)](https://github.com/user-attachments/assets/4e36278f-1271-409e-8fb4-8325e10e7d57)

#### Step 22: Visualization of the Correlation using Heatmap.
![Screenshot (46)](https://github.com/user-attachments/assets/5c2de34c-fab3-407d-9b6f-bc88db0e6670)

### FINAL RESULT

![Screenshot (48)](https://github.com/user-attachments/assets/51a3aed1-6454-4dbf-90d0-c9371777c36f)

### OUTLIER DATA

#### Step 23: Assigning variable for numeric type columns excluding the object columns.
![Screenshot (49)](https://github.com/user-attachments/assets/b9956d0c-b47e-4d53-98ff-3ad293acf653)

#### Step 24: Calculation of Outliers using some criterias
![Screenshot (51)](https://github.com/user-attachments/assets/d69cd1f8-d3a6-49c4-bb93-c75f66b140df)

### DATA TRANSFORMATION

#### Step 25: Grouping Object and Unobject Datatypes in a variable
![Screenshot (52)](https://github.com/user-attachments/assets/d3983238-f24e-4723-bcb5-9a3c2e10a778)

#### Step 26: Fitting Data using LabelEncoder
![Screenshot (53)](https://github.com/user-attachments/assets/3d2dd671-d048-42e0-a05c-265ed3e5deea)

#### Step 27: Merging the Object Data and Unobject Data together and a preview of our Dataset
![Screenshot (54)](https://github.com/user-attachments/assets/ed8f42f9-1d99-45e8-be5f-6189c82b0807)

### CREATING MODELS

#### Step 28: Splitting Dataset into Features (X) and Target (Y)
![Screenshot (55)](https://github.com/user-attachments/assets/27bf6203-2b51-42ae-af0d-863bb2f4733d)

#### Step 29: Using train_test_split model for splitting
![Screenshot (56)](https://github.com/user-attachments/assets/1b1c8806-abb2-4335-bdc8-2658c9bef2f2)

#### Step 30: Models for Checking and Testing Accuracy Score
![Screenshot (57)](https://github.com/user-attachments/assets/6a1253ff-ad81-4580-b8b2-41f6406a7d3d)

#### Step 31: Defining a function for our Models
![Screenshot (58)](https://github.com/user-attachments/assets/6d8c023e-1de0-41cb-af04-72ab0630de79)

#### Step 32: Displaying the MODEL SCORE
![Screenshot (59)](https://github.com/user-attachments/assets/e7f7c845-270b-4588-8fb9-07b6879b41bd)

#### Step 33: Passing the Model Score into a DataFrame
![Screenshot (60)](https://github.com/user-attachments/assets/9f0238ce-b672-4854-be5f-d9a10c37adfe)

#### Step 34: Plotting the Random Mean Squared Error Model Score using Line plot
![Screenshot (61)](https://github.com/user-attachments/assets/dc5ce4f3-c833-4d93-b630-314ac9c0dabc)

#### Step 35: Plotting the R2 Model Score using Line plot
![Screenshot (62)](https://github.com/user-attachments/assets/dfdd53bd-f53e-4f78-8c28-a9ec4de5007b)

### SHOWING THE DIFFERENCES BETWEEN THE ACTUAL VALUES AND THE PREDICTED VALUES USING MODEL3 - RandomForestRegressor Model
![Screenshot (63)](https://github.com/user-attachments/assets/518a2062-db1a-4c68-b94a-1b43fba90ba2)

# THANK YOU







