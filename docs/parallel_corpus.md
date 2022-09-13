# Parallel Corpus

This is a guide how to obtain the parallel corpus used in the paper of TransCoder-ST.

## Download Java Files from Google BigQuery
- Create a Google platform account (you will have around 300 $ given for free, that is sufficient for GitHub)
- Create a Google BigQuery project
- In this project, create a dataset
- In this dataset, create one table per programming language. The results of each SQL request (one per language) will be stored in these tables.
- Before running your SQL request, make sure you change the query settings to save the query results in the dedicated table (More -> Query Settings -> Destination -> table for query results -> put table name)
- Run your SQL request (one per language and dont forget to change the table for each request)
- Export your results to google Cloud:
  - In google cloud storage, create a bucket and a folder per language into it
  - Export your table to this bucket (EXPORT -> Export to GCS -> export format JSON, compression GZIP)
- To download the bucket on your machine, use the API gsutil:
  - pip install gsutil
  - gsutil config -> to config gsutil with your google account
  - gsutil -m cp -r gs://name_of_bucket/name_of_folder . -> copy your bucket on your machine

More information about the GitHub dataset on BigQuery can be found [here](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code).

Example query to obtain all Java files:

```sql
WITH selected_repos as (
    SELECT f.id, f.repo_name as repo_name, f.ref as ref, f.path as path, l.license
    FROM `bigquery-public-data.github_repos.files` as f
    JOIN `bigquery-public-data.github_repos.licenses` as l on l.repo_name = f.repo_name
    WHERE l.license = 'mit' OR l.license = 'apache-2.0'
),
deduped_files as (
    SELECT f.id, MIN(f.repo_name) as repo_name, MIN(f.ref) as ref, MIN(f.path) as path, MIN(f.license) as license
    FROM selected_repos as f
    GROUP BY f.id
)
SELECT
    f.repo_name,
    f.ref,
    f.path,
    f.license,
    c.copies,
    c.content,
FROM deduped_files as f
JOIN `bigquery-public-data.github_repos.contents` as c on f.id = c.id
WHERE
    NOT c.binary
    AND f.path like '%.java'
```

Note that this will only include the repositories with `mit` or `aache-2.0` license. Contrary Rozière et al. used `mit`, `apache-2.0`, `gpl-2.0`, `gpl-3.0`, `bsd-2-clause` and `bsd-3-clause` licenses.

If you want to obtain the files for C++ and Python, just exchange `%.java` with `%.cpp` or `%.py` respectively.

Unzip the downloaded Java files and put it under `data/java_functions`.

## Preprocess Java Files
Run the following script extract the standalone Java functions, tokenize and apply BPE encoding:

```sh
codegen_sources/scripts/transcoder_st/preprocess.sh
```

Note this can take a very long time. Consider splitting it up in multiple chunks or use a lot of CPUs for processing.

## Create Parallel Corpus
Run the following script to create the automated unit test for the standalone Java functions, translate the unit test to C++ and Python and to create the final parallel corpus:

```sh
cg/codegen_sources/scripts/transcoder_st/create_dataset.sh
```

Note this can take a very long time. Consider splitting it up in multiple chunks or use a lot of CPUs for processing.