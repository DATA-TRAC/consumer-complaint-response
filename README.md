# Consumer Complaint Response

## Project Description

* This project aims to predict a company's response to a complaint made by a consumer to the Consumer Financial Protection Bureau to see if the wording of a complaint can give a better or worse response from the company.

## Project Goal

* This analysis aims to provide an accurate prediction of company response based on the language of a complaint.

### Initial Thoughts

* There are going to be keywords that match a company's response.
* Sentiment analysis will not be useful because most complaints will likely be negative.

## The Plan

### Acquire

* Data acquired from [Google BigQuery](https://console.cloud.google.com/marketplace/product/cfpb/complaint-database)
* 3458906 rows × 18 columns *before* cleaning

### Prepare

* **<span style="color:red">Dropped Columns</span>**
  * **product**
    * 0% nulls
    * *<span style="color:orange">ENGINEERED FEATURE</span>*
    * **<span style="color:orange">bin related products/services together then drop</span>**
        * <span style="color:orange">bins = credit_report, credit_card, debt_collection, mortgage, bank, loans, and money_service</span>
  * **subproduct**
    * 7% null
  * **issue**
    * 0% nulls
    * 165 unique values
    * **<span style="color:blue">future iteration</span>**
  * **subissue**
    * 20% null
    * 221 unique
  * **consumer_complaint_narrative**
    * 64% null
    * renamed to narrative
      * **<span style="color:orange">drop all null values</span>**
      * **<span style="color:orange">drop after NLTK cleaning</span>**
  * **company_public_response**
    * 56% null
    * related to target
  * **zip code**
    * 1% null
    * mixed data types
      * **<span style="color:blue">future iteration</span>**
  * **consumer_consent_provided**
    * 25% null
    * does not relate to the target
  * **submitted_via**
    * 0% nulls
    * does not relate to the target
  * **date_sent_to_company**
    * 0% nulls
      * **<span style="color:blue">future iteration</span>**
  * **timely_response**
    * 0% nulls
    * boolean
      * **<span style="color:blue">future iteration</span>**
  * **consumer_disputed**
    * 77% null
      * **<span style="color:blue">future iteration</span>**
  * **complaint_id**
    * 0% nulls
<br>
* **<span style="color:green">Cleaned Columns</span>**
  * **date_received**
    * 0% nulls
    * changed date to DateTime
  * **company_name**
    * 0% nulls
    * 6,694 Companies
  * **state**
    * 1% null
    * keep for purposes of exploration
      * **<span style="color:orange">impute 1% null into UNKNOWN label</span>**
  * **tags**
    * 89% null
        * **<span style="color:orange">impute nulls with "Average Person label</span>**
  * **company_response_to_consumer**
    * Target
    * 4 nulls = 0%
      * **<span style="color:orange">drop these 4 rows because this is the target column</span>**
    * 8 initial unique values
      * **<span style="color:blue">nice to have: apply the model to in_progress complaints and see what it predicts based on the language</span>**
      * **<span style="color:orange">drop 'in progress' response because there is no conclusion</span>**
---

### Post Cleaning Inspection

1246736 rows x 8 columns

Used NLTK to clean each document resulting in:
* 2 new columns: *clean* (removes redacted XXs, and stopwords removed) and *lemon* (lemmatized)
    <br>
    <br>
    
Selected columns to proceed with after cleaning:
* date_received, product_bins, company_name, state, tags, company_response_to_customer (target), clean, lemon

---

### Explore

**1. Are there words that get particular responses and is there a relationship?**
* What are the payout words that got a company response of closed with monetary relief?
* Are there unique words associated with products? Is there a relationship between unique product words and responses?
<br>

**2. Do all responses have a negative sentiment?**
* Do narratives with a neutral or positive sentiment analysis relating to bank account products lead to a response of closed with monetary relief?
<br>

**3. Are there unique words associated with the most negative and most positive company responses?**
<br>

**4. Which product is more likely to have monetary relief?**
      
      
*****************************************    


**SECOND ITERATION QUESTIONS**

5. Is there a relationship/bias for servicemember tags in relation to company response?
      * good but it's better after MVP
      NEW - Is the company response independent from a consumer being a service member or not?
          - Is the company response independent from a consumer having a tag?

7. Do some companies proportionally give better or worse responses (relief/no relief)?
      * Look at this after MVP
6. Does narrative length relate to company response?
      * plain answer - may be out
2. Do consumers in some states receive monetary relief more frequently than others?
      * Look at this after MVP
9. Are there more complaints during certain seasons of the year?
      * not useful for modeling - OUT
                
                
## Data Dictionary

| Feature                               | Definition                                                                                  |
| :------------------------------------ | :------------------------------------------------------------------------------------------ |
| date_received                         | Date the complaint was received by the CFPB                                                 |
| product                               | The type of product the consumer identified in the complaint                                |
| subproduct                            | The type of sub-product the consumer identified in the complaint                            |
| issue                                 | The issue the consumer identified in the complaint                                          |
| subissue                              | The sub-issue the consumer identified in the complaint                                      |
| consumer_complaint_narrative          | A description of the complaint provided by the consumer                                     |
| company_public_response               | The company's optional public-facing response to a consumer's complaint                     |
| company_name                          | Name of the company identified in the complaint by the consumer                             |
| state                                 | Two-letter postal abbreviation of the state of the mailing address provided by the consumer |
| zip_code                              | The mailing ZIP code provided by the consumer                                               |
| tags                                  | Older American is aged 62 and older, Servicemember is Active/Guard/Reserve member or spouse |
| consumer_consent_provided             | Identifies whether the consumer opted in to publish their complaint narrative               |
| submitted_via                         | How the complaint was submitted to the CFPB                                                 |
| date_sent_to_company                  | The date the CFPB sent the complaint to the company                                         |
| company_response_to_consumer (target) | The response from the company about this complaint                                          |
| timely_response                       | Indicates whether the company gave a timely response or not                                 |
| consumer_disputed                     | Whether the consumer disputed the company's response                                        |
| complaint_id                          | Unique ID for complaints registered with the CFPB                                           |
| product_bins                          | Engineered Feature: bin related products together                                           |
| clean                                 | Engineered Feature: tokenized, numbers/specials, and XX's removed                           |
| lemon                                 | Engineered Feature: clean column PLUS lemmatization                                         |


## Model
    - Decission Tree
    - KNN
    - Multinomial
    - 



## Steps to Reproduce

1) Clone this repo
   * You may need to update your Python Libraries, my libraries were updated on 5 June 2023 for this project
2) For a quick run
   * Verify `import wrangle as w` is in the imports section of final_notebook
   * Run final_notebook
   * This will use a pre-built and cleaned dataset based on the datasets from the longer run in step 3
3) For the longer run
   * ⚠️WARNING⚠️: These are almost the same steps I took to originally acquire the data. The steps take a lot of time (and space) and may not even be the best way of doing it. I highly recommend doing the quick run in step 2 unless you want to know how I got the data.
   * Verify `import big_wrangle as w` is in the imports section of final_notebook
   * Install the pandas-gbq package
     * `conda install pandas-gbq --channel conda-forge`
     * `pip install pandas-gbq`
   * Go to Google BigQuery and create a project
     * Copy and run the 'long-SQL queries found in `big_wrangle.py` in [Google BigQuery](https://cloud.google.com/bigquery/public-data)
       * Click on 'Go to Datasets in Cloud Marketplace' and search for 'CFPB' or 'complain' and view the dataset to open a quick SQL prompt to query in
     * Save each result as a BigQuery table in your project
       * You can look in `big_wrangle.py` for what I named my project, database, and tables
     * Edit and save the 'small-SQL query variables found in `big_wrangle.py` to the respective table names in your BigQuery project using this format: `FROM 'database. table'` and edit the 'project_ID' variable to your project's ID
   * Run final_notebook
     * It may ask for authentication when it tries to query Google BigQuery
     * Try to run again if it stopped for authentication
   * This will run through the longer pathway of getting the datasets from the source and merging/cleaning/prep
     * It will probably take a while (millions of rows, +2GB), hence I do not recommend

* **Quick run**
    * Verify `import wrangle as w` is in the imports section 
    * Run final report
    * This will use a pre-built and cleaned parquet file
<br>
<br>
* **For the longer run: ⚠️WARNING⚠️:** These are almost the same steps we took to originally acquire the data. The steps take a lot of time (and space) and may not even be the best way of doing it. We highly recommend doing the quick run above unless you want to know how we got the data.
    * Verify `import big_wrangle as w` is in the imports section
    * Install the pandas-gbq package
        * `pip install pandas-gbq`
    * Go to Google BigQuery and create a project
    * Copy the `'long-SQL queries found in big_wrangle.py`
        * Run in [Google BigQuery](https://cloud.google.com/bigquery/public-data)
    * Click on 'Go to Datasets in Cloud Marketplace' and search for 'CFPB'
        * View the dataset to open a quick SQL prompt to query in
    * Save each result as a BigQuery table in your project
    * You can look in `big_wrangle.py for what we named our project, database, and tables`
    * Edit and save the `'small-SQL query variables found in big_wrangle.py` to the respective table names in your BigQuery project using this format: 
        * ***FROM 'database. table' and edit the 'project_ID' variable to your project's ID***
    * Run final report
    * It may ask for authentication when it tries to query Google BigQuery
        * Try to run again if it stopped
    * This will run through the longer pathway of getting the datasets from the source and merging/cleaning/prep
    * It will probably take a while **(3+ millions of rows, +2GB)**

---

## Explore Takeaways

## Recommendations/Next Steps

* Nice-To-Haves: Second iteration looking at discrimination based on zip code/state and company response, applying the model to in_progress complaints and see what it predicts based on the language after company response, 16 SVB complaints -- *can possibly add as an end project application/impact to identify fraudulent activity or discrimination based on customer complaints
