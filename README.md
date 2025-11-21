# mlflow-evals

## Usage
The flow to the evaluation is to prepare the inputs with valid manual tests (with more or less info, steps and possible extra files).
Then run "create_dataset_from_yaml.py" that will loop through all files in the "datasets" directory and:
- Create one dataset for each entry and uploads it to MLFLow
- If files are present it will upload the files to the SembiIQ API (and replace the doc_ids in the dataset before uploading)

You will be able to check the datasets in MLFlow running localy (localhost:5500).


### 1. Docker containers
Start docker container with MLFLow and PostgreSQL with the following command:
```
docker-compose up
```
### 2. Prepare inputs
Two directories are available to add files "datasets" and "new_datasets".
Files present in there have all information necessary to create the tests in SembiIQ (plus the expected outputs) and will upload all that info to MLFlow.
Add files with tests that you want to achieve.

### 3. Upload tests to MLFLow
Once all files are created run:
```
python3 create_dataset_from_yaml.py
```

This will create a new experiment in MLFLow and upload al the datasets to it.

### 4. Run evals
In this steps you must have already created the datasets (if not please check Step 2 and 3 above).
The following command will process all registered datasets and send them to SembiIQ. Register the response in MLFLow within the Experiment under Evaluation runs.
```
python3 run_generation_from_dataset.py
```

