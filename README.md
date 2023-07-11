# Consumer Complaint Response

## Project Description

* This project aims to predict a company's response to a complaint made by a consumer to the Consumer Financial Protection Bureau to see if the wording of a complaint can give a better or worse response from the company.

## Project Goal

* This analysis aims to provide an accurate prediction of company response based on the language of a complaint.

### Initial Thoughts

* There are going to be key words that match to a companies response.

## The Plan

### Acquire

  * Data acquired from [Google BigQuery](https://console.cloud.google.com/marketplace/product/cfpb/complaint-database)
  * 3458906 rows × 18 columns *before* cleaning
  
### Prepare

* date_receieved 
    * changed date to datetime
    * no nulls
    * 2011 to 2023
* product 
    * no nulls
    * credit related
        * **bin related services together**
* subproduct
    * 7% null
    * top value = credit reporting
    * fill nulls with the product
    * what does subproduct correlate with?
        * **drop entirely for the MVP**
* issue
    * no nulls
    * 165 unique values
    * concat into consumer_complaint_narrative column to address those nulls and then drop issue
        * **issue, subissue, and narrative are all consumer entered items so we are not risking unethical manipulation of the data because the source is the consumer**
* subissue
    * 20% null
    * 221 unique
        * **concat and drop with issue INTO narrative**
* consumer_complaint_narrative (ENGINEERED FEATURE)
    * 64% null before imputing above values
        * narrative_plus_issue
            * **AFTER concat = 0% null
* company_public_response
    * 56% null
        * **drop company_public_response because it doesn't relate to the target or even features**
* company_name
    * no nulls
    * 6,694 companies
        * **16 SVB complaints -- *can possibly add as an end project application/impact to identify fraudlent activity or discrimination based on customer complaints***
        * **SPICY**
* state
    * 1% null
    * based on per capita, population, and state size... 
    * keep for purposes of exploration
    * do not bin (causes manipulation)
        * **bin 1% null into UNKNOWN labeling**
* zip code
    * 1% null
    * located a string buried in the data
        * **use re to clean**
        * **drop for MVP, nice-to-have for second iteration looking at discrimination**
* tags
    * 89% null
    * domain knowledge: 62 and older accounted for senior - pulled straight from source
        * **impute nulls with "Average Person**
* consumer_consent_provided
    * does not relate to target
        * **drop**
* submitted_via
    * no nulls
        * **drop because imbalanced data, doesn't provide enough value for target**
* date_sent_to_company
    * no nulls
        * **drop because no value and we also dropped submitted_via which includes mail, fax, etc...**
* company_response_to_consumer
    * 4 nulls = 0%
        * **drop these 4 rows because this is the target column**
    * 8 unique values
        * **investigate the difference between closed without relief and closed with relief**
        * **NICE_TO_HAVE: applying model to in_progess complaints and see what it predicts based on the language**
    * 7 unique values after dropping 'in_progress'
* timely_response
    * no nulls
    * boolean
        * **drop because it is noise, no value to target**
* consumer_disputed
    * 77% null
        * **drop because this data has a lot of nulls**
* complaint_id
    * no nulls
        * **drop, not valuable for MVP. Can be used for nice_to_haves**

---

**Post Univariate Inspection**

* 3355342 rows x 7 columns *after* cleaning
    * date_recieved, product, narrative_plus_issue, compnay_name, state, tags, company_response_to_customer (target)

---

### Explore

  * Questions
    1.
    2.
    3.
    4.

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
| state                                 | Two letter postal abbreviation of the state of the mailing address provided by the consumer |
| zip_code                              | The mailing ZIP code provided by the consumer                                               |
| tags                                  | Older American is aged 62 and older, Servicemember is Active/Guard/Reserve member or spouse |
| consumer_consent_provided             | Identifies whether the consumer opted in to publish their complaint narrative               |
| submitted_via                         | How the complaint was submitted to the CFPB                                                 |
| date_sent_to_company                  | The date the CFPB sent the complaint to the company                                         |
| company_response_to_consumer (target) | The response from the company about this complaint                                          |
| timely_response                       | Indicates whether the company gave a timely response or not                                 |
| consumer_disputed                     | Whether the consumer disputed the company's response                                        |
| complaint_id                          | Unique ID for complaints registered with the CFPB                                           |
|                                       |                                                                                             |

## Steps to Reproduce

1) Clone this repo
   *  You may need to update your Python Libraries, my libraries were updated on 5 June, 2023 for this project
2) For a quick run
   * Verify `import wrangle as w` is in the imports section of final_notebook
   * Run final_notebook
   * This will use a pre-built and cleaned dataset based off of the datasets from the longer run in step 3
3) For the longer run
   * ⚠️WARNING⚠️: These are almost the same steps I took to originally acquire the data. The steps take a lot of time (and space) and may not even be the best way of doing it. I highly recommend to do the quick run in step 2 unless you want to know how I got the data.
   * Verify `import big_wrangle as w` is in the imports section of final_notebook
   * Install the pandas-gbq package
     * `conda install pandas-gbq --channel conda-forge`
     * `pip install pandas-gbq`
   * Go to Google BigQuery and create a project
     * Copy and run the 'long-sql' queries found in `big_wrangle.py` in [Google BigQuery](https://cloud.google.com/bigquery/public-data)
       * Click on 'Go to Datasets in Cloud Marketplace' and search for 'CFPB' or 'complain' and view the dataset to open a quick sql prompt to query in
     * Save each result as a BigQuery table in your project
       * You can look in `big_wrangle.py` for what I named my project, database, and tables
     * Edit and save the 'small-sql' query variables found in `big_wrangle.py` to the respective table names in your BigQuery project using this format: `FROM 'database.table'` and edit the 'project_ID' variable to your project's ID
   * Run final_notebook
     * It may ask for authentication when it tries to query Google BigQuery
     * Try to run again if it stopped for authentication
   * This will run through the longer pathway of getting the datasets from the source and merge/clean/prep
     * It will probably take awhile (millions of rows, +2GB), hence I do not recommend

---

## Explore Takeaways

## Recommendations/Next Steps

* insert
* insert
* insert
