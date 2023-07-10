# `Title`

# Project Description

* This repo contains a project ___ Using analysis and data aggregating techniques we were able to answer the questions listed below ___ The deliverables are ___

# Project Goal

* This analysis aims to address the listed questions and uncover any additional important findings related to the data.

# Initial Thoughts

* There are going to be valuable insights that we can use  to ___

# The Plan

* Acquire
    * Data acquired from [Google BigQuery](https://console.cloud.google.com/marketplace/product/cfpb/complaint-database)
    * Joined ___ created a CSV ___
    * ___ rows × __ columns *before* cleaning
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

# Data Dictionary  

| Feature | Definition|
|:--------|:-----------|
|date_received|Date the complaint was received by the CPFB|
|product|The type of product the consumer identified in the complaint|
|subproduct|The type of sub-product the consumer identified in the complaint|
|issue|The issue the consumer identified in the complaint|
|subissue|The sub-issue the consumer identified in the complaint|
|consumer_complaint_narrative|A description of the complaint provided by the consumer|
|company_public_response|The company's optional public-facing response to a consumer's complaint|
|company_name|Name of the company identified in the complaint by the consumer|
|state|Two letter postal abbreviation of the state of the mailing address provided by the consumer|
|zip_code|The mailing ZIP code provided by the consumer|
|tags|Data that supports easier searching and sorting of complaints|
|consumer_consent_provided|Identifies whether the consumer opted in to publish their complaint narrative|
|submitted_via|How the complaint was submitted to the CFPB|
|date_sent_to_company|The date the CFPB sent the complaint to the company|
|company_response_to_consumer (target)|The response from the company about this complaint|
|timely_response|Indicates whether the company gave a timely response or not|
|consumer_disputed|Whether the consumer disputed the company's response|
|complaint_id|Unique ID for complaints registered with the CFPB|
|| |

# Steps to Reproduce
1. Clone this repo
2. Run notebook

---

# Explore Takeaways

1.
  
2.

3.
 
4.
   
5.
 
# Recommendations/Next Steps

* insert

* insert

* insert