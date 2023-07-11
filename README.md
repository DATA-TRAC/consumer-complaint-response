# Consumer Complaint Response

## Project Description

* This repo contains a project ___Using analysis and data aggregating techniques we were able to answer the questions listed below___ The deliverables are ___

## Project Goal

* This analysis aims to address the listed questions and uncover any additional important findings related to the data.

### Initial Thoughts

* There are going to be valuable insights that we can use  to ___

## The Plan

* Acquire

  * Data acquired from [Google BigQuery](https://console.cloud.google.com/marketplace/product/cfpb/complaint-database)
  * Joined ___created a CSV___
  * 3458906 rows × 18 columns *before* cleaning
  * ___ rows × __ columns *after* cleaning
  
* Prepare

  * dropped columns
    * insert
  * renamed columns
    * insert
    * insert
    * insert
  * changed dates to datetime type
  * created new columns
    * insert
    * insert
    * insert
    * insert
  * no nulls
* Explore

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
