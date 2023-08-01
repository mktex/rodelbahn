### Usage:
1. Run the following commands in the project's root directory to set up
   database and model.
-  To run ETL pipeline that cleans data and stores in database:
<pre>
    python ./data/process_data.py ./data/disaster_messages.csv  ./data/disaster_categories.csv data/DisasterResponse.db
</pre>

-  To run ML pipeline that trains classifier and saves python (this
   might require several minutes to complete)
<pre>
    python ./models/train_classifier.py ./data/DisasterResponse.db ./data/rodelbahn_model.pckl
</pre>

2. Run the following command in the app's directory to run your web app.
<pre>   python ./app/run.py </pre>

3. In Browser http://locahost:3001/
