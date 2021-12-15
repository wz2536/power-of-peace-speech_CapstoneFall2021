# The Power of Peace Speech

Columbia University - Data Science Institute - FALL 2021 Capstone Project

**Team Members**<br>
Hongling Liu (hl3418@columbia.edu)<br>
Haoyue Qi (hq2180@columbia.edu)<br>
Xuanhao Wu (xw2744@columbia.edu)<br>
Yuxin Zhou (yz3904@columbia.edu)<br>
Wenjie Zhu (wz2536@columbia.edu)

**Advisors**\
Peter T. Coleman, Professor, Psychology and Education<br>
Allegra Chen-Carrel, Program Manager, The Sustaining Peace Project<br>
Philippe Loustaunau, Managing Partner, Vista Consulting LLC<br>
Larry S. Liebovitch, Professor, Physics and Psychology


## Project Overview

In the current world of rising conflicts and wars, peacekeepers are actively conducting research on hate speech analysis in seeking to maintain robust and peaceful communities. Such hate speech dataset is accessible on the internet including but not limited to news articles, blogs and social media posts. Meanwhile, the peace speech, which is also an important yet available piece of language assets, commonly gets neglected. Capstone students in the past years had worked with Professor Coleman’s team to conduct pilot studies that showed promising powers of the peace speech in distinguishing the peacefulness of societies. This year, we will continue the previous research with a much richer dataset and two new approaches:

1) Explore the performance of more state-of-arts natural language processing (NLP) models on classifying the text embedding of English articles from high-peace and low-peace countries produced by BERT.
2) Study how the interrelationship among words affect the previous classification results by evaluating the models’ performance over randomly shuffled articles.

### Data
The data from LexisNexis and is stored on the AWS S3 bucket as .json files in two separate directories: highPeace and lowPeace. Each directory contains news articles from the world-wise top 10 countries of each category based on the rule-of-5 metric. Here is a full list of countries: 

lowPeace | highPeace |
--------- | --------- |
Afghanistan| Austria |
Congo| Australia | 
Guinea | Belgium |
India | Czech Republic |
Iran | Denmark |
Kenya | Finland |
Nigeria | Netherlands |
Sri Lanka | New Zealand |
Uganda | Norway |
Zimbabwe | Sweden |

Each .json file contains the data related to one article. There are ~57M articles in lowPeace and ~33M articles (.json files) in highPeace.

### Files and Folder
To reduce the waiting time of loading data multiple times in to notebook memory, we combine some tasks that are closely related to a step in one notebook and perform these tasks sequentially, instead of starting a new notebook for each task in all steps. The detailed content per notebook is described as below.

- **Reports** contains our progress reports and final presentation slides of the projects.
- **Sample Data Exploration** contains the works we done in progress report 1; construct theoretical approach to the problem based on the sample dataset
- **Data EDA** contains the processes of reading in all seperate .json files, compressing them into large .csv files, uploading the files to S3, and performing EDA.
- **Preprocess** contains the codes for preprocessing the .csv stored in S3 from **Data EDA** step, performing train/test split, and storing the result back to S3 as .csv and .json format.
- **Classification Models** contains all files related to model training experiments and model performance evaluation

